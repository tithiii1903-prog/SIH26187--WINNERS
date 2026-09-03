"""
Real-Time Frame Processing Pipeline.

Processes video frames sequentially from a VideoSource, applying AI modules
(person detection, tracking, vehicle detection, virtual fence)
and producing annotated frames for MJPEG streaming.

Architecture (Producer / AI Worker / Stream Pipeline):
    VideoSource
        ↓
    Capture & Display Thread (Native source FPS ~20–30 FPS)
        ↓ (bounded size-1 slot, drops stale frames)
    AI Processing Worker (Async ~8–15 FPS)
        ↓
    Thread-Safe AI State Snapshot (per-track isolation & grace smoothing)
        ↓
    Annotation on Capture Frames
        ↓
    MJPEG Stream Buffer (Smooth, responsive, zero-lag)
"""

import cv2
import time
import threading
import numpy as np
from collections import deque
from typing import Dict, Any, List, Optional, Set, Tuple

from .video_source import VideoSource
from .detector import PersonDetector
from .tracker import PersonTracker
from .vehicle_detector import VehicleDetector
from .virtual_fence import VirtualFence
from .. import database


def _calc_measured_fps(timestamps: deque, window_sec: float = 1.5) -> float:
    """Calculate actual measured FPS over a rolling time window."""
    now = time.time()
    while timestamps and timestamps[0] < now - window_sec:
        timestamps.popleft()
    if len(timestamps) < 2:
        return 0.0
    duration = timestamps[-1] - timestamps[0]
    if duration <= 0:
        return 0.0
    return (len(timestamps) - 1) / duration


class FrameProcessor:
    """
    Real-time concurrent frame processor.

    Design Principles:
    - Display/Stream thread NEVER blocks on heavy AI inference (YOLO, DB).
    - Single-slot bounded latest-frame buffer between Capture and AI worker.
    - Stale frames are dropped if AI falls behind to guarantee zero latency accumulation.
    - Strict per-track isolation for tracking and virtual fence intrusions.
    - Time-based grace smoothing to prevent track destruction on transient missed detections.
    - True measured FPS telemetry for Source, AI, and Stream.
    """

    MAX_EVENTS = 500
    TRACK_EXPIRY_SEC = 1.2  # Expire stale tracks not refreshed by AI within 1.2s

    def __init__(
        self,
        video_source: VideoSource,
        person_detector: PersonDetector,
        person_tracker: PersonTracker,
        vehicle_detector: VehicleDetector,
        virtual_fence: VirtualFence,
        fence_config_path: str = "config/zones.json",
    ):
        self.video_source = video_source
        self.person_detector = person_detector
        self.person_tracker = person_tracker
        self.vehicle_detector = vehicle_detector
        self.virtual_fence = virtual_fence
        self.fence_config_path = fence_config_path

        # Threading & Control
        self._capture_thread: Optional[threading.Thread] = None
        self._ai_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._ai_frame_event = threading.Event()

        # Dedicated fine-grained locks
        self._ai_buffer_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._display_lock = threading.Lock()
        self._modules_lock = threading.Lock()

        # Single-slot bounded AI input buffer
        self._pending_ai_frame: Optional[np.ndarray] = None
        self._pending_ai_frame_idx: int = 0
        self._pending_ai_timestamp: float = 0.0

        # Latest annotated display frame
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_ready = threading.Event()

        # Module toggles (all ON by default)
        self._modules_enabled = {
            "human_detection": True,
            "human_tracking": True,
            "vehicle_detection": True,
            "virtual_fence": True,
        }

        # Immutable latest AI state snapshot (swapped atomically by AI worker)
        self._latest_ai_state: Dict[str, Any] = {
            "persons": {},  # track_id -> dict
            "vehicles": [],
            "active_intrusions": {},  # track_id -> zone_id
            "ai_fps": 0.0,
            "timestamp": 0.0,
            "frame_idx": 0,
        }

        # Measured FPS rolling windows
        self._source_timestamps: deque = deque(maxlen=60)
        self._ai_timestamps: deque = deque(maxlen=60)
        self._stream_timestamps: deque = deque(maxlen=60)

        # Measured telemetry values
        self._measured_source_fps = 0.0
        self._measured_ai_fps = 0.0
        self._measured_stream_fps = 0.0

        # Live analytics snapshot (read by API)
        self._analytics: Dict[str, Any] = {
            "current_persons": 0,
            "active_tracks": 0,
            "peak_persons": 0,
            "current_vehicles": 0,
            "peak_vehicles": 0,
            "cars": 0,
            "motorcycles": 0,
            "buses": 0,
            "trucks": 0,
            "active_intrusions": [],
            "total_intrusion_entries": 0,
            "total_intrusion_exits": 0,
            "processing_fps": 0.0,
            "stream_fps": 0.0,
            "source_fps": 0.0,
            "frames_processed": 0,
            "max_active_tracks": 0,
            "timestamp": 0.0,
        }

        # Bounded events deque
        self._events: deque = deque(maxlen=self.MAX_EVENTS)

        # State & lifecycle
        self._status = "READY"  # READY, STARTING, LIVE, STOPPING, STOPPED, ERROR
        self._error_message: Optional[str] = None
        self._fence_reload_requested = False

    # ------------------------------------------------------------------
    # Public Properties & Control API
    # ------------------------------------------------------------------

    @property
    def status(self) -> str:
        with self._state_lock:
            return self._status

    @property
    def error_message(self) -> Optional[str]:
        with self._state_lock:
            return self._error_message

    def get_modules(self) -> Dict[str, bool]:
        with self._modules_lock:
            return self._modules_enabled.copy()

    def set_modules(self, modules: Dict[str, bool]):
        with self._modules_lock:
            for key, value in modules.items():
                if key in self._modules_enabled:
                    self._modules_enabled[key] = value

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest annotated frame (single-slot buffer, thread-safe)."""
        with self._display_lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    def wait_for_frame(self, timeout: float = 1.0) -> bool:
        return self._frame_ready.wait(timeout=timeout)

    def get_analytics(self) -> Dict[str, Any]:
        """Get current live analytics snapshot."""
        with self._state_lock:
            return self._analytics.copy()

    def get_events(self, since_index: int = 0) -> List[Dict[str, Any]]:
        with self._state_lock:
            events_list = list(self._events)
            if 0 < since_index < len(events_list):
                return events_list[since_index:]
            return events_list

    def reload_fence(self):
        """Signal that the virtual fence polygon should be reloaded from DB."""
        with self._state_lock:
            self._fence_reload_requested = True

    def start(self):
        """Start both Capture & Stream and AI Worker background threads with fresh session state."""
        if self._capture_thread is not None and self._capture_thread.is_alive():
            return

        self._stop_event.clear()
        self._ai_frame_event.clear()
        self._frame_ready.clear()

        with self._state_lock:
            self._status = "STARTING"
            self._error_message = None
            self._events.clear()
            self._fence_reload_requested = False

        # Reset transient AI buffer & display frame
        with self._ai_buffer_lock:
            self._pending_ai_frame = None
            self._pending_ai_frame_idx = 0
            self._pending_ai_timestamp = 0.0

        with self._display_lock:
            self._latest_frame = None

        # Reset model and tracking states
        if hasattr(self.person_detector, "reset"):
            self.person_detector.reset()
        if hasattr(self.person_tracker, "reset"):
            self.person_tracker.reset()
        if hasattr(self.virtual_fence, "reset"):
            self.virtual_fence.reset()

        # Reset FPS windows & telemetry
        self._source_timestamps.clear()
        self._ai_timestamps.clear()
        self._stream_timestamps.clear()
        self._measured_source_fps = 0.0
        self._measured_ai_fps = 0.0
        self._measured_stream_fps = 0.0

        self._latest_ai_state = {
            "persons": {},
            "vehicles": [],
            "active_intrusions": {},
            "ai_fps": 0.0,
            "timestamp": 0.0,
            "frame_idx": 0,
        }

        self._analytics = {
            "current_persons": 0,
            "active_tracks": 0,
            "peak_persons": 0,
            "current_vehicles": 0,
            "peak_vehicles": 0,
            "cars": 0,
            "motorcycles": 0,
            "buses": 0,
            "trucks": 0,
            "active_intrusions": [],
            "total_intrusion_entries": 0,
            "total_intrusion_exits": 0,
            "processing_fps": 0.0,
            "stream_fps": 0.0,
            "source_fps": 0.0,
            "frames_processed": 0,
            "max_active_tracks": 0,
            "timestamp": 0.0,
        }

        # Start AI worker thread first
        self._ai_thread = threading.Thread(target=self._ai_worker_loop, daemon=True, name="AIWorkerThread")
        self._ai_thread.start()

        # Start Capture & Display thread
        self._capture_thread = threading.Thread(target=self._capture_and_stream_loop, daemon=True, name="CaptureStreamThread")
        self._capture_thread.start()

    def stop(self):
        """Signal all threads to stop and release resources cleanly."""
        with self._state_lock:
            if self._status in ("STOPPED", "READY"):
                pass
            else:
                self._status = "STOPPING"

        self._stop_event.set()
        self._ai_frame_event.set()
        self._frame_ready.set()

        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=3.0)
        self._capture_thread = None

        if self._ai_thread is not None and self._ai_thread.is_alive():
            self._ai_thread.join(timeout=3.0)
        self._ai_thread = None

        if hasattr(self.video_source, "close"):
            self.video_source.close()
        elif hasattr(self.video_source, "release"):
            self.video_source.release()

        with self._ai_buffer_lock:
            self._pending_ai_frame = None

        with self._display_lock:
            self._latest_frame = None

        with self._state_lock:
            self._status = "STOPPED"

    # ------------------------------------------------------------------
    # AI Worker Loop (Thread 2 — Async AI Inference)
    # ------------------------------------------------------------------

    def _ai_worker_loop(self):
        """Asynchronously processes latest camera frames through AI modules."""
        while not self._stop_event.is_set():
            # Wait until a new frame is dispatched from capture thread
            if not self._ai_frame_event.wait(timeout=0.1):
                continue

            # Retrieve newest frame from bounded slot
            with self._ai_buffer_lock:
                if self._pending_ai_frame is None:
                    self._ai_frame_event.clear()
                    continue
                frame_to_process = self._pending_ai_frame
                frame_idx = self._pending_ai_frame_idx
                timestamp = self._pending_ai_timestamp
                self._pending_ai_frame = None
                self._ai_frame_event.clear()

            modules = self.get_modules()
            new_events = []

            # 1. Hot-reload virtual fence if requested
            with self._state_lock:
                reload_fence = self._fence_reload_requested
                if reload_fence:
                    self._fence_reload_requested = False
            if reload_fence:
                self.virtual_fence.reload_zones()

            # 2. Person Detection & ByteTrack
            person_detections = []
            new_track_ids: List[int] = []
            disappeared_track_ids: List[int] = []

            if modules["human_detection"]:
                try:
                    if modules["human_tracking"]:
                        raw_dets = self.person_detector.track(frame_to_process)
                        person_detections, new_track_ids, disappeared_track_ids = self.person_tracker.update_trajectories(
                            raw_dets, frame_idx, timestamp
                        )
                    else:
                        person_detections = self.person_detector.detect(frame_to_process)
                except Exception as e:
                    print(f"[FrameProcessor AI] Person detection error: {e}")
                    person_detections = []

            # 3. Track Lifecycle Events (NEW_TRACK / TRACK_DISAPPEARED)
            if modules["human_detection"] and modules["human_tracking"]:
                for tid in new_track_ids:
                    desc = f"Track ID {tid} entered the frame."
                    try:
                        database.insert_event(
                            event_type="NEW_TRACK",
                            timestamp=timestamp,
                            message=desc,
                            track_id=tid
                        )
                    except Exception as e:
                        print(f"Failed to log NEW_TRACK event: {e}")
                    new_events.append({
                        "timestamp": round(timestamp, 2),
                        "type": "NEW_TRACK",
                        "track_id": tid,
                        "description": desc,
                    })

                for tid in disappeared_track_ids:
                    desc = f"Track ID {tid} left the frame."
                    try:
                        database.insert_event(
                            event_type="TRACK_DISAPPEARED",
                            timestamp=timestamp,
                            message=desc,
                            track_id=tid
                        )
                    except Exception as e:
                        print(f"Failed to log TRACK_DISAPPEARED event: {e}")
                    new_events.append({
                        "timestamp": round(timestamp, 2),
                        "type": "TRACK_DISAPPEARED",
                        "track_id": tid,
                        "description": desc,
                    })

            # 4. Virtual Fence Intrusions
            active_intrusions = {}
            if modules["virtual_fence"] and modules["human_detection"]:
                try:
                    fence_events = self.virtual_fence.process_frame(
                        person_detections, timestamp
                    )
                    new_events.extend(fence_events)
                    active_intrusions = self.virtual_fence.get_active_intrusions()
                except Exception as e:
                    print(f"[FrameProcessor AI] Virtual fence error: {e}")
                    active_intrusions = {}

            # 5. Vehicle Detection
            vehicle_detections = []
            if modules["vehicle_detection"]:
                try:
                    vehicle_detections = self.vehicle_detector.detect(frame_to_process)
                except Exception as e:
                    print(f"[FrameProcessor AI] Vehicle detection error: {e}")
                    vehicle_detections = []

            # 6. Measure AI Performance
            now = time.time()
            self._ai_timestamps.append(now)
            self._measured_ai_fps = _calc_measured_fps(self._ai_timestamps)

            # 7. Construct Immutable Per-Track AI State Snapshot with Grace Smoothing
            persons_map: Dict[Any, Dict[str, Any]] = {}
            current_active_ids = set()

            for det in person_detections:
                tid = det.get("id")
                box = det["box"]
                conf = det["conf"]
                traj = det.get("trajectory", [])

                if tid is not None:
                    current_active_ids.add(tid)

                is_intruding = bool(
                    tid is not None
                    and tid in active_intrusions
                    and modules["virtual_fence"]
                )

                p_data = {
                    "id": tid,
                    "box": box,
                    "conf": conf,
                    "trajectory": traj,
                    "intrusion": is_intruding,
                    "last_seen_time": now,
                    "last_seen_frame": frame_idx,
                }

                if tid is not None:
                    persons_map[tid] = p_data
                else:
                    persons_map[f"untracked_{len(persons_map)}"] = p_data

            # AI Snapshot Grace Smoothing:
            # If YOLO briefly misses detections (transient empty frame),
            # preserve active tracks from tracker that are still within grace period.
            if len(persons_map) == 0 and modules["human_tracking"] and hasattr(self.person_tracker, "get_active_tracks_snapshot"):
                active_snap = self.person_tracker.get_active_tracks_snapshot(now=now)
                for tid, p_info in active_snap.items():
                    current_active_ids.add(tid)
                    is_intruding = bool(
                        tid in active_intrusions and modules["virtual_fence"]
                    )
                    p_info_copy = p_info.copy()
                    p_info_copy["intrusion"] = is_intruding
                    persons_map[tid] = p_info_copy

            v_counts = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0}
            for v_det in vehicle_detections:
                cn = v_det.get("class_name")
                if cn in v_counts:
                    v_counts[cn] += 1

            intrusion_entries = sum(1 for e in new_events if e["type"] == "INTRUSION_ENTER")
            intrusion_exits = sum(1 for e in new_events if e["type"] == "INTRUSION_EXIT")

            num_persons = len(persons_map)
            num_vehicles = len(vehicle_detections)

            # 8. Atomically Swap State Snapshot & Update Analytics
            new_ai_snapshot = {
                "persons": persons_map,
                "vehicles": vehicle_detections,
                "active_intrusions": active_intrusions.copy(),
                "ai_fps": round(self._measured_ai_fps, 1),
                "timestamp": timestamp,
                "frame_idx": frame_idx,
            }

            with self._state_lock:
                self._latest_ai_state = new_ai_snapshot

                self._analytics["current_persons"] = num_persons
                self._analytics["active_tracks"] = len(current_active_ids)
                if num_persons > self._analytics["peak_persons"]:
                    self._analytics["peak_persons"] = num_persons
                self._analytics["current_vehicles"] = num_vehicles
                if num_vehicles > self._analytics["peak_vehicles"]:
                    self._analytics["peak_vehicles"] = num_vehicles
                self._analytics["cars"] = v_counts["Car"]
                self._analytics["motorcycles"] = v_counts["Motorcycle"]
                self._analytics["buses"] = v_counts["Bus"]
                self._analytics["trucks"] = v_counts["Truck"]
                self._analytics["active_intrusions"] = list(active_intrusions.keys())
                self._analytics["total_intrusion_entries"] += intrusion_entries
                self._analytics["total_intrusion_exits"] += intrusion_exits
                self._analytics["processing_fps"] = round(self._measured_ai_fps, 1)
                self._analytics["frames_processed"] = frame_idx
                self._analytics["timestamp"] = round(timestamp, 2)
                if len(current_active_ids) > self._analytics["max_active_tracks"]:
                    self._analytics["max_active_tracks"] = len(current_active_ids)

                for ev in new_events:
                    self._events.append(ev)

    # ------------------------------------------------------------------
    # Capture & Display Loop (Thread 1 — High-FPS Frame Producer)
    # ------------------------------------------------------------------

    def _capture_and_stream_loop(self):
        """Reads frames at source FPS, overlays latest AI state, updates MJPEG buffer."""
        try:
            if not self.video_source.open():
                with self._state_lock:
                    self._status = "ERROR"
                    self._error_message = "Failed to open video source"
                return

            source_fps = self.video_source.fps if hasattr(self.video_source, "fps") else self.video_source.get_fps()
            frame_interval = 1.0 / source_fps if source_fps > 0 else 1.0 / 30.0

            with self._state_lock:
                self._status = "LIVE"
                self._analytics["source_fps"] = source_fps

            frames_captured = 0
            source_read_failed = False

            while not self._stop_event.is_set():
                loop_start = time.time()

                success, raw_frame = self.video_source.read_frame()
                if not success:
                    source_read_failed = True
                    break

                frames_captured += 1
                now = time.time()
                self._source_timestamps.append(now)
                self._measured_source_fps = _calc_measured_fps(self._source_timestamps)
                if self._measured_source_fps <= 0:
                    self._measured_source_fps = source_fps

                timestamp = frames_captured / source_fps if source_fps > 0 else 0.0

                # 1. Push newest frame into bounded AI slot (replaces older pending frame)
                with self._ai_buffer_lock:
                    self._pending_ai_frame = raw_frame.copy()
                    self._pending_ai_frame_idx = frames_captured
                    self._pending_ai_timestamp = timestamp
                    self._ai_frame_event.set()

                # 2. Get quick copy of latest AI state snapshot & modules
                with self._state_lock:
                    ai_snapshot = self._latest_ai_state
                    ai_fps = self._analytics.get("processing_fps", 0.0)

                modules = self.get_modules()
                annotated_frame = raw_frame.copy()

                # 3. Draw Virtual Fence Zones
                if modules["virtual_fence"] and modules["human_detection"]:
                    for zone in self.virtual_fence.get_zones():
                        if not zone.get("enabled", True):
                            continue
                        polygon = np.array(zone["polygon"], np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [polygon], True, (0, 0, 255), 2)
                        cv2.putText(
                            annotated_frame, zone["name"],
                            tuple(zone["polygon"][0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                        )

                # 4. Draw Tracked Persons (Strict Per-Track Isolation)
                active_persons_count = 0
                if modules["human_detection"]:
                    for tid, person in ai_snapshot["persons"].items():
                        # Skip stale tracks that haven't been seen recently
                        if now - person.get("last_seen_time", 0.0) > self.TRACK_EXPIRY_SEC:
                            continue

                        active_persons_count += 1
                        box = person["box"]
                        conf = person["conf"]
                        track_id = person.get("id")
                        is_intruding = person.get("intrusion", False)

                        # Bounding Box Color Logic:
                        # - Normal person -> ALWAYS GREEN (0, 255, 0)
                        # - Fence intruding -> RED (0, 0, 255)
                        if is_intruding:
                            box_color = (0, 0, 255)  # RED
                            thickness = 3
                        else:
                            box_color = (0, 255, 0)  # GREEN
                            thickness = 2

                        cv2.rectangle(
                            annotated_frame,
                            (box[0], box[1]), (box[2], box[3]),
                            box_color, thickness,
                        )

                        # Person / Track ID Label
                        if track_id is not None and modules["human_tracking"]:
                            label = f"Track ID: {track_id} ({conf:.2f})"
                        else:
                            label = f"Person {conf:.2f}"
                        if is_intruding:
                            label += " - INTRUSION"

                        (lw, lh), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                        )
                        cv2.rectangle(
                            annotated_frame,
                            (box[0], box[1] - lh - baseline - 2),
                            (box[0] + lw + 4, box[1]),
                            box_color, cv2.FILLED,
                        )
                        cv2.putText(
                            annotated_frame, label,
                            (box[0] + 2, box[1] - baseline - 1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255) if is_intruding else (0, 0, 0), 1,
                        )

                        # Trajectory Motion Trail & Tracking Dots
                        if modules["human_tracking"]:
                            if "trajectory" in person and len(person["trajectory"]) > 0:
                                traj = person["trajectory"]
                                for i in range(len(traj)):
                                    cv2.circle(annotated_frame, traj[i], 3, (0, 255, 255), -1)
                                    if i > 0:
                                        cv2.line(annotated_frame, traj[i - 1], traj[i], (0, 255, 255), 2)
                                # Prominent dot at current position
                                cv2.circle(annotated_frame, traj[-1], 5, (0, 165, 255), -1)
                                cv2.circle(annotated_frame, traj[-1], 2, (0, 0, 255), -1)
                            else:
                                # Center dot for immediate single-frame detection
                                cx = (box[0] + box[2]) // 2
                                cy = (box[1] + box[3]) // 2
                                cv2.circle(annotated_frame, (cx, cy), 5, (0, 255, 255), -1)

                # 5. Draw Vehicles
                if modules["vehicle_detection"]:
                    v_colors = {
                        "Car": (255, 128, 0),
                        "Motorcycle": (0, 128, 255),
                        "Bus": (255, 0, 255),
                        "Truck": (128, 255, 0),
                    }
                    for v_det in ai_snapshot["vehicles"]:
                        v_box = v_det["box"]
                        v_conf = v_det["conf"]
                        class_name = v_det["class_name"]
                        color = v_colors.get(class_name, (0, 255, 0))

                        cv2.rectangle(
                            annotated_frame,
                            (v_box[0], v_box[1]), (v_box[2], v_box[3]),
                            color, 2,
                        )
                        v_label = f"[{class_name}] {v_conf:.2f}"
                        (lw, lh), baseline = cv2.getTextSize(
                            v_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                        )
                        cv2.rectangle(
                            annotated_frame,
                            (v_box[0], v_box[1] - lh - baseline),
                            (v_box[0] + lw, v_box[1]),
                            color, cv2.FILLED,
                        )
                        cv2.putText(
                            annotated_frame, v_label,
                            (v_box[0], v_box[1] - baseline),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                        )

                # 6. Measure Stream FPS
                self._stream_timestamps.append(now)
                self._measured_stream_fps = _calc_measured_fps(self._stream_timestamps)

                # 7. Draw Real Measured Performance HUD
                cv2.putText(
                    annotated_frame,
                    f"AI FPS: {ai_fps:.1f} | STREAM FPS: {self._measured_stream_fps:.1f} | SOURCE FPS: {self._measured_source_fps:.1f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
                )
                cv2.putText(
                    annotated_frame,
                    f"Persons: {active_persons_count}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
                )
                cv2.putText(
                    annotated_frame,
                    f"Vehicles: {len(ai_snapshot['vehicles'])}",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 128, 0), 2
                )

                hud_y = 130
                active_intrusions = ai_snapshot.get("active_intrusions", {})
                if active_intrusions and modules["virtual_fence"]:
                    cv2.putText(
                        annotated_frame,
                        f"ACTIVE INTRUSIONS: {len(active_intrusions)}",
                        (20, hud_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2,
                    )

                # 8. Store annotated frame in display slot
                with self._display_lock:
                    self._latest_frame = annotated_frame

                with self._state_lock:
                    self._analytics["stream_fps"] = round(self._measured_stream_fps, 1)

                self._frame_ready.set()
                self._frame_ready.clear()

                # 9. Pacing for simulated-live file playback
                elapsed = time.time() - loop_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    self._stop_event.wait(timeout=sleep_time)

            with self._state_lock:
                if self._status == "LIVE":
                    if source_read_failed and self.video_source.is_live():
                        self._status = "ERROR"
                        self._error_message = "Camera disconnected or unavailable"
                    else:
                        self._status = "STOPPED"

        except Exception as e:
            with self._state_lock:
                self._status = "ERROR"
                self._error_message = str(e)
            import traceback
            traceback.print_exc()

        finally:
            if hasattr(self.video_source, "close"):
                self.video_source.close()
            elif hasattr(self.video_source, "release"):
                self.video_source.release()
            self._stop_event.set()
            self._ai_frame_event.set()
            self._frame_ready.set()
            with self._display_lock:
                self._latest_frame = None
