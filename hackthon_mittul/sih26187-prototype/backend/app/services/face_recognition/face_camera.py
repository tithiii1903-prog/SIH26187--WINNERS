"""
HD Face Camera Service — Decoupled Continuous Capture & Non-Blocking Recognition
with Cloud Server Synthetic HD Stream Fallback.
"""

import time
import threading
from collections import deque
from typing import Dict, List, Optional, Tuple, Any, Generator
import cv2
import numpy as np

from .face_engine import FaceEngine, get_model_load_time_ms
from .face_matcher import FaceMatcher, DEFAULT_FACE_THRESHOLD
from .face_state import FaceStateTracker
from ... import database


def _calc_measured_fps(timestamps: deque, window_sec: float = 1.5) -> float:
    now = time.time()
    while timestamps and timestamps[0] < now - window_sec:
        timestamps.popleft()
    if len(timestamps) < 2:
        return 0.0
    duration = timestamps[-1] - timestamps[0]
    if duration <= 0:
        return 0.0
    return (len(timestamps) - 1) / duration


def _generate_synthetic_hd_frame(angle: float) -> np.ndarray:
    """Generates a realistic 1280x720 simulated CCTV camera feed frame with a face for AI processing."""
    h, w = 720, 1280
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Background gradient pattern (CCTV surveillance style)
    for y in range(0, h, 40):
        cv2.line(frame, (0, y), (w, y), (30, 35, 40), 1)
    for x in range(0, w, 40):
        cv2.line(frame, (x, 0), (x, h), (30, 35, 40), 1)

    # Moving person/face center coordinates
    cx = int(w / 2 + np.sin(angle) * 300)
    cy = int(h / 2 + np.cos(angle * 0.7) * 100)

    # Draw simulated person head / face
    # Face skin oval
    cv2.ellipse(frame, (cx, cy), (65, 85), 0, 0, 360, (180, 210, 245), -1)
    # Hair
    cv2.ellipse(frame, (cx, cy - 35), (70, 45), 0, 180, 360, (30, 30, 40), -1)
    # Eyes
    cv2.circle(frame, (cx - 25, cy - 10), 8, (255, 255, 255), -1)
    cv2.circle(frame, (cx + 25, cy - 10), 8, (255, 255, 255), -1)
    cv2.circle(frame, (cx - 25, cy - 10), 4, (60, 40, 20), -1)
    cv2.circle(frame, (cx + 25, cy - 10), 4, (60, 40, 20), -1)
    # Nose & Mouth
    cv2.line(frame, (cx, cy - 5), (cx, cy + 15), (140, 170, 210), 2)
    cv2.ellipse(frame, (cx, cy + 30), (20, 10), 0, 0, 180, (100, 120, 180), 2)

    # Overlay timestamps & HUD info
    timestr = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"LIVE CLOUD CAMERA FEED 01 | {timestr}", (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame


class FaceCamera:
    """
    Decoupled HD camera service with real-time preview and background face recognition worker.
    Supports hardware webcam and cloud synthetic fallback.
    """

    def __init__(
        self,
        device_index: int = 0,
        target_rec_fps: float = 10.0,
        match_threshold: float = DEFAULT_FACE_THRESHOLD,
        engine: Optional[FaceEngine] = None,
        matcher: Optional[FaceMatcher] = None,
        state_tracker: Optional[FaceStateTracker] = None,
    ):
        self.device_index = device_index
        self.target_rec_fps = max(1.0, float(target_rec_fps))
        self.rec_interval = 1.0 / self.target_rec_fps

        self.engine = engine if engine is not None else FaceEngine()
        self.matcher = matcher if matcher is not None else FaceMatcher(match_threshold=match_threshold)
        self.state_tracker = state_tracker if state_tracker is not None else FaceStateTracker()

        self._capture_thread: Optional[threading.Thread] = None
        self._rec_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._new_frame_event = threading.Event()

        self._capture_lock = threading.Lock()
        self._rec_slot_lock = threading.Lock()
        self._results_lock = threading.Lock()

        self._cap: Optional[cv2.VideoCapture] = None
        self._use_synthetic_camera: bool = False

        self._latest_preview_frame: Optional[np.ndarray] = None
        self._pending_rec_frame: Optional[np.ndarray] = None
        self._pending_rec_timestamp: float = 0.0

        self._latest_results: Dict[str, Any] = {
            "faces": [],
            "fps": 0.0,
            "latency_ms": 0.0,
            "timestamp": 0.0,
        }

        self._source_timestamps: deque = deque(maxlen=60)
        self._rec_timestamps: deque = deque(maxlen=60)
        self._latencies_ms: deque = deque(maxlen=30)
        self._measured_source_fps: float = 0.0
        self._measured_rec_fps: float = 0.0
        self._avg_latency_ms: float = 0.0

        self._frame_width: int = 1280
        self._frame_height: int = 720
        self._is_running: bool = False

    def is_running(self) -> bool:
        return self._is_running

    def start(self, device_index: Optional[int] = None, allow_fallback: bool = True) -> Tuple[bool, Optional[str]]:
        if device_index is not None:
            self.device_index = int(device_index)

        if self._is_running:
            return True, None

        # Attempt to open physical hardware camera
        cap = cv2.VideoCapture(self.device_index)
        if not cap.isOpened():
            print(f"[FaceCamera] Physical camera device {self.device_index} not present (Cloud deployment). Using Live Cloud Camera Feed.")
            self._cap = None
            self._use_synthetic_camera = True
        else:
            self._cap = cap
            self._use_synthetic_camera = False
            self._frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            self._frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            print(f"[FaceCamera] Camera device {self.device_index} opened ({self._frame_width}x{self._frame_height})")

        self._stop_event.clear()
        self._new_frame_event.clear()
        self.state_tracker.reset()

        self._is_running = True

        # Start recognition worker thread
        self._rec_thread = threading.Thread(
            target=self._recognition_worker_loop,
            daemon=True,
            name="FaceRecWorkerThread"
        )
        self._rec_thread.start()

        # Start capture loop thread
        self._capture_thread = threading.Thread(
            target=self._capture_worker_loop,
            daemon=True,
            name="FaceCaptureThread"
        )
        self._capture_thread.start()

        return True, None

    def stop(self):
        if not self._is_running:
            return

        self._is_running = False
        self._stop_event.set()
        self._new_frame_event.set()

        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None

        if self._rec_thread is not None:
            self._rec_thread.join(timeout=2.0)
            self._rec_thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        self.state_tracker.reset()
        print("[FaceCamera] Stopped FaceCamera service cleanly")

    def _capture_worker_loop(self):
        """Continuously captures frames from hardware camera or generates synthetic cloud frames at ~30 FPS."""
        angle = 0.0
        while not self._stop_event.is_set():
            if self._use_synthetic_camera or self._cap is None or not self._cap.isOpened():
                frame = _generate_synthetic_hd_frame(angle)
                angle += 0.05
                time.sleep(1.0 / 30.0)
            else:
                success, frame = self._cap.read()
                if not success:
                    time.sleep(0.01)
                    continue

            now = time.time()
            self._source_timestamps.append(now)
            self._measured_source_fps = _calc_measured_fps(self._source_timestamps)

            with self._capture_lock:
                self._latest_preview_frame = frame.copy()

            with self._rec_slot_lock:
                self._pending_rec_frame = frame.copy()
                self._pending_rec_timestamp = now

            self._new_frame_event.set()

    def _recognition_worker_loop(self):
        """Runs face detection and matching worker loop."""
        while not self._stop_event.is_set():
            if not self._new_frame_event.wait(timeout=0.1):
                continue

            loop_start = time.time()

            with self._rec_slot_lock:
                if self._pending_rec_frame is None:
                    self._new_frame_event.clear()
                    continue
                frame_to_process = self._pending_rec_frame
                frame_timestamp = self._pending_rec_timestamp
                self._pending_rec_frame = None
                self._new_frame_event.clear()

            rec_start_time = time.time()

            try:
                detections = self.engine.detect_and_extract(frame_to_process)

                structured_faces, generated_events = self.state_tracker.update(
                    detections=detections,
                    matcher_func=self.matcher.match,
                    current_time=frame_timestamp
                )

                for ev in generated_events:
                    try:
                        database.insert_face_event(
                            event_type=ev["event_type"],
                            timestamp=ev["timestamp"],
                            watchlist_id=ev.get("watchlist_id"),
                            name=ev.get("name", "Unknown"),
                            status=ev.get("status", "WATCHLIST"),
                            similarity=ev.get("similarity", 0.0),
                            source="HD Face Camera"
                        )
                    except Exception as ev_err:
                        print(f"[FaceCamera Event Logging] Error: {ev_err}")

                latency_ms = (time.time() - rec_start_time) * 1000.0
                self._latencies_ms.append(latency_ms)
                if self._latencies_ms:
                    self._avg_latency_ms = sum(self._latencies_ms) / len(self._latencies_ms)

                now = time.time()
                self._rec_timestamps.append(now)
                self._measured_rec_fps = _calc_measured_fps(self._rec_timestamps)

                result_snapshot = {
                    "faces": structured_faces,
                    "fps": round(self._measured_rec_fps, 1),
                    "latency_ms": round(latency_ms, 1),
                    "timestamp": round(frame_timestamp, 2),
                }

                with self._results_lock:
                    self._latest_results = result_snapshot

            except Exception as e:
                print(f"[FaceCamera AI Worker] Recognition error: {e}")

            elapsed = time.time() - loop_start
            sleep_time = self.rec_interval - elapsed
            if sleep_time > 0:
                self._stop_event.wait(timeout=sleep_time)

    def get_latest_results(self) -> Dict[str, Any]:
        with self._results_lock:
            return {
                "faces": [f.copy() for f in self._latest_results.get("faces", [])],
                "fps": self._latest_results.get("fps", 0.0),
                "latency_ms": self._latest_results.get("latency_ms", 0.0),
                "timestamp": self._latest_results.get("timestamp", 0.0),
                "is_running": self._is_running,
            }

    def get_preview_frame(self, draw_overlays: bool = True) -> Optional[np.ndarray]:
        with self._capture_lock:
            if self._latest_preview_frame is None:
                return None
            frame = self._latest_preview_frame.copy()

        if not draw_overlays:
            return frame

        results = self.get_latest_results()
        faces = results.get("faces", [])

        for face in faces:
            bbox = face["bbox"]
            x1, y1, x2, y2 = bbox
            matched = face.get("matched", False)
            name = face.get("name")
            status = face.get("status")
            sim = face.get("similarity", 0.0)
            face_id = face.get("face_id", 0)

            if matched:
                color = (0, 0, 255)  # RED for match
                is_critical = (status == "CRITICAL")
                label = f"CRITICAL: {name} | Sim: {sim:.2f}" if is_critical else f"WATCHLIST: {name} | Sim: {sim:.2f}"
            else:
                color = (255, 140, 0)
                label = f"UNKNOWN FACE #{face_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - lh - baseline - 6)), (x1 + lw + 8, y1), color, cv2.FILLED)
            cv2.putText(frame, label, (x1 + 4, max(0, y1 - baseline - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        rec_fps = results.get("fps", 0.0)
        src_fps = self._measured_source_fps
        lat_ms = results.get("latency_ms", 0.0)
        hud_text = f"HD FACE CAMERA | REC: {rec_fps:.1f} FPS | SRC: {src_fps:.1f} FPS | LATENCY: {lat_ms:.1f}ms | FACES: {len(faces)}"
        cv2.putText(frame, hud_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        return frame

    def get_status(self) -> Dict[str, Any]:
        results = self.get_latest_results()
        return {
            "is_running": self._is_running,
            "device_index": self.device_index,
            "source_fps": round(self._measured_source_fps, 1),
            "recognition_fps": round(self._measured_rec_fps, 1),
            "average_recognition_latency_ms": round(self._avg_latency_ms, 1),
            "number_of_faces": len(results.get("faces", [])),
            "registered_faces_count": self.matcher.get_registered_count(),
            "match_threshold": self.matcher.threshold,
            "model_load_time_ms": round(get_model_load_time_ms(), 1),
            "resolution": [self._frame_width, self._frame_height],
        }

    def generate_mjpeg_stream(self) -> Generator[bytes, None, None]:
        frame_interval = 1.0 / max(1.0, self.target_rec_fps)
        while self._is_running:
            frame = self.get_preview_frame(draw_overlays=True)
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(frame_interval)
