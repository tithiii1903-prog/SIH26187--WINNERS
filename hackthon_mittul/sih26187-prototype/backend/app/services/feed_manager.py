"""
Feed Manager — Camera/Feed Registry and Lifecycle Management.

Manages feed metadata (CRUD), file uploads, and the single active processing session.
Only ONE feed can be LIVE at a time (memory constraint for 16GB Apple Silicon).
Feed metadata is persisted in config/feeds.json.
"""

import os
import json
import uuid
import cv2
import shutil
import threading
from typing import Dict, Optional, Any

from .video_source import FileVideoSource, CameraVideoSource
from .frame_processor import FrameProcessor
from .detector import PersonDetector
from .tracker import PersonTracker
from .vehicle_detector import VehicleDetector
from .virtual_fence import VirtualFence


FEEDS_CONFIG_PATH = "config/feeds.json"
UPLOADS_DIR = "uploads"
ZONES_CONFIG_PATH = "config/zones.json"
MAX_UPLOAD_SIZE_BYTES = 200 * 1024 * 1024  # 200 MB


class FeedManager:
    """
    Manages feed metadata, uploads, and the single active processing session.

    Enforces: only ONE feed can be LIVE at a time.
    """

    def __init__(self):
        self._feeds: Dict[str, Dict[str, Any]] = {}
        self._active_processor: Optional[FrameProcessor] = None
        self._active_feed_id: Optional[str] = None
        self._lock = threading.Lock()

        # Shared AI model instances (loaded once, reused across feeds)
        self._person_detector: Optional[PersonDetector] = None
        self._vehicle_detector: Optional[VehicleDetector] = None

        # Ensure directories exist
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        os.makedirs("config", exist_ok=True)

        # Load persisted feeds
        self._load_feeds()

    def _load_feeds(self):
        """Load feed metadata from disk."""
        if os.path.exists(FEEDS_CONFIG_PATH):
            try:
                with open(FEEDS_CONFIG_PATH, "r") as f:
                    data = json.load(f)
                    self._feeds = data.get("feeds", {})
                    # Reset all statuses to READY on startup
                    for feed_id, feed in self._feeds.items():
                        if feed["status"] in ("LIVE", "STARTING", "STOPPING"):
                            feed["status"] = "STOPPED"
            except Exception as e:
                print(f"Failed to load feeds config: {e}")
                self._feeds = {}

    def _save_feeds(self):
        """Persist feed metadata to disk."""
        try:
            with open(FEEDS_CONFIG_PATH, "w") as f:
                json.dump({"feeds": self._feeds}, f, indent=2)
        except Exception as e:
            print(f"Failed to save feeds config: {e}")

    def _ensure_models_loaded(self):
        """Load AI models if not already loaded (shared instances)."""
        if self._person_detector is None:
            print("Loading PersonDetector model...")
            self._person_detector = PersonDetector("yolov8n.pt")
            print(f"PersonDetector loaded on: {self._person_detector.device}")
        if self._vehicle_detector is None:
            print("Loading VehicleDetector model...")
            self._vehicle_detector = VehicleDetector("yolov8n.pt")
            print(f"VehicleDetector loaded on: {self._vehicle_detector.device}")

    def _cleanup_finished_processor(self):
        """Auto-cleanup processor if it has finished (end of video or error)."""
        if self._active_processor and self._active_processor.status in ("STOPPED", "ERROR"):
            proc_status = self._active_processor.status
            if self._active_feed_id and self._active_feed_id in self._feeds:
                self._feeds[self._active_feed_id]["status"] = proc_status
            self._active_processor = None
            self._active_feed_id = None
            self._save_feeds()

    def list_feeds(self) -> list:
        """Return all feeds as a list."""
        with self._lock:
            self._cleanup_finished_processor()
            feeds = []
            for feed_id, feed in self._feeds.items():
                feed_copy = feed.copy()
                if feed_id == self._active_feed_id and self._active_processor:
                    feed_copy["status"] = self._active_processor.status
                feeds.append(feed_copy)
            return feeds

    def get_feed(self, feed_id: str) -> Optional[Dict[str, Any]]:
        """Get a single feed's metadata."""
        with self._lock:
            self._cleanup_finished_processor()
            feed = self._feeds.get(feed_id)
            if feed is None:
                return None
            feed_copy = feed.copy()
            if feed_id == self._active_feed_id and self._active_processor:
                feed_copy["status"] = self._active_processor.status
            return feed_copy

    def create_feed(self, name: str, filepath: str, filename: str) -> Dict[str, Any]:
        """
        Create a new feed from an uploaded/existing file.
        Validates the video file with OpenCV.
        """
        # Validate with OpenCV
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            cap.release()
            raise ValueError("Failed to open video file with OpenCV")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else None
        cap.release()

        feed_id = str(uuid.uuid4())[:8]
        feed = {
            "id": feed_id,
            "name": name,
            "source_type": "file",
            "filename": filename,
            "filepath": filepath,
            "width": width,
            "height": height,
            "fps": round(fps, 2),
            "frame_count": frame_count,
            "duration": round(duration, 2) if duration else None,
            "status": "READY",
        }

        with self._lock:
            self._feeds[feed_id] = feed
            self._save_feeds()

        return feed

    def create_camera_feed(self, name: str, device_index: int = 0) -> Dict[str, Any]:
        """
        Register a device camera as a feed.
        Probes the camera to validate availability and read resolution/FPS.
        The camera is released immediately after probing — it will be
        re-opened when the feed is started.
        """
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == 'nt' else [cv2.CAP_ANY]
        cap = None
        for backend in backends:
            try:
                temp_cap = cv2.VideoCapture(device_index, backend)
                if temp_cap.isOpened():
                    ret, frame = temp_cap.read()
                    if ret and frame is not None and frame.size > 0:
                        cap = temp_cap
                        break
                    temp_cap.release()
            except Exception:
                pass

        if cap is None or not cap.isOpened():
            width, height, fps = 1280, 720, 30.0
        else:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()

        feed_id = str(uuid.uuid4())[:8]
        feed = {
            "id": feed_id,
            "name": name,
            "source_type": "camera",
            "device_index": device_index,
            "filename": f"Device {device_index}",
            "filepath": "",
            "width": width,
            "height": height,
            "fps": round(fps, 2),
            "frame_count": None,
            "duration": None,
            "status": "READY",
        }

        with self._lock:
            self._feeds[feed_id] = feed
            self._save_feeds()

        return feed

    def delete_feed(self, feed_id: str) -> bool:
        """Delete a feed and its uploaded file."""
        with self._lock:
            self._cleanup_finished_processor()
            if feed_id not in self._feeds:
                return False

            # Cannot delete feed that is actively processing
            if feed_id == self._active_feed_id and self._active_processor:
                if self._active_processor.status in ("LIVE", "STARTING"):
                    raise ValueError("Cannot delete feed that is currently active. Stop it first.")

            feed = self._feeds[feed_id]
            filepath = feed.get("filepath", "")

            # Remove file if in uploads directory (file-based feeds only)
            if filepath and filepath.startswith(UPLOADS_DIR) and os.path.exists(filepath):
                os.remove(filepath)

            del self._feeds[feed_id]
            self._save_feeds()
            return True

    def start_feed(self, feed_id: str) -> Dict[str, Any]:
        """Start processing a feed. Stops any currently active feed first."""
        with self._lock:
            if feed_id not in self._feeds:
                raise ValueError(f"Feed {feed_id} not found")

            feed = self._feeds[feed_id]

        # Stop any currently active feed
        if self._active_feed_id is not None and self._active_feed_id != feed_id:
            self.stop_feed(self._active_feed_id)

        if self._active_processor:
            if self._active_processor.status == "LIVE" and self._active_feed_id == feed_id:
                raise ValueError("Feed is already live")
            # If stopped/error/starting, clean up completely before restarting
            self._active_processor.stop()
            self._active_processor = None
            self._active_feed_id = None

        # Ensure models are loaded
        self._ensure_models_loaded()

        # Reset detector model tracking state
        if self._person_detector is not None:
            self._person_detector.reset()

        # Create the appropriate video source based on source type
        source_type = feed.get("source_type", "file")
        if source_type == "camera":
            video_source = CameraVideoSource(feed.get("device_index", 0))
        else:
            video_source = FileVideoSource(feed["filepath"])

        person_tracker = PersonTracker(max_history=30, grace_period_sec=0.8, max_inactive_frames=15)
        virtual_fence = VirtualFence()

        processor = FrameProcessor(
            video_source=video_source,
            person_detector=self._person_detector,
            person_tracker=person_tracker,
            vehicle_detector=self._vehicle_detector,
            virtual_fence=virtual_fence,
            fence_config_path=ZONES_CONFIG_PATH,
        )

        with self._lock:
            self._active_processor = processor
            self._active_feed_id = feed_id
            self._feeds[feed_id]["status"] = "STARTING"
            self._save_feeds()

        processor.start()

        return self.get_feed(feed_id)

    def stop_feed(self, feed_id: str) -> Dict[str, Any]:
        """Stop processing a feed."""
        with self._lock:
            if feed_id != self._active_feed_id:
                raise ValueError("Feed is not currently active")

        if self._active_processor:
            self._active_processor.stop()

        with self._lock:
            self._feeds[feed_id]["status"] = "STOPPED"
            self._active_processor = None
            self._active_feed_id = None
            self._save_feeds()

        return self.get_feed(feed_id)

    def get_active_processor(self) -> Optional[FrameProcessor]:
        """Get the currently active frame processor."""
        with self._lock:
            return self._active_processor

    def get_active_feed_id(self) -> Optional[str]:
        """Get the currently active feed ID."""
        with self._lock:
            return self._active_feed_id

    def reload_fence(self):
        """Signal the active processor to reload the virtual fence polygon."""
        with self._lock:
            if self._active_processor:
                self._active_processor.reload_fence()
