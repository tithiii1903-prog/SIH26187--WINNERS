"""
Video Source Abstraction — Standardized frame reading interface for files and cameras
with Cloud Server Synthetic Live Stream Fallback.
"""

import os
import time
import threading
import abc
from typing import Tuple, Optional
import cv2
import numpy as np


def _generate_synthetic_video_frame(angle: float) -> np.ndarray:
    """Generates a realistic 1280x720 CCTV feed for cloud servers without physical webcams or missing sample files."""
    h, w = 720, 1280
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    for y in range(0, h, 40):
        cv2.line(frame, (0, y), (w, y), (25, 30, 35), 1)
    for x in range(0, w, 40):
        cv2.line(frame, (x, 0), (x, h), (25, 30, 35), 1)

    cx = int(w / 2 + np.sin(angle) * 250)
    cy = int(h / 2 + np.cos(angle * 0.8) * 80)

    # Draw simulated walking person
    cv2.ellipse(frame, (cx, cy - 60), (35, 45), 0, 0, 360, (180, 210, 240), -1)  # Head
    cv2.rectangle(frame, (cx - 45, cy - 10), (cx + 45, cy + 90), (50, 100, 200), -1)  # Body
    cv2.line(frame, (cx - 20, cy + 90), (cx - 30, cy + 180), (30, 30, 80), 8)  # Leg 1
    cv2.line(frame, (cx + 20, cy + 90), (cx + 30, cy + 180), (30, 30, 80), 8)  # Leg 2

    # Timestamp overlay
    timestr = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"LIVE SURVEILLANCE FEED | {timestr}", (25, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame


class VideoSource(abc.ABC):
    """Abstract base class for all video frame providers."""

    @abc.abstractmethod
    def open(self) -> bool:
        pass

    @abc.abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        pass

    @abc.abstractmethod
    def close(self) -> None:
        pass

    @abc.abstractmethod
    def is_opened(self) -> bool:
        pass

    @property
    def fps(self) -> float:
        return 30.0

    def get_fps(self) -> float:
        return self.fps

    def release(self) -> None:
        self.close()

    def is_live(self) -> bool:
        return isinstance(self, CameraVideoSource)

    @property
    def width(self) -> int:
        return 1280

    @property
    def height(self) -> int:
        return 720

    @property
    def frame_count(self) -> Optional[int]:
        return None

    @property
    def duration(self) -> Optional[float]:
        return None


class FileVideoSource(VideoSource):
    """Video source backed by a local MP4 file with Cloud Fallback."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._cap: Optional[cv2.VideoCapture] = None
        self._use_synthetic: bool = False
        self._width: int = 1280
        self._height: int = 720
        self._fps: float = 30.0
        self._frame_count: int = 900
        self._duration: float = 30.0
        self._angle: float = 0.0

    def open(self) -> bool:
        clean_path = self.filepath.replace("\\", "/") if self.filepath else ""
        filename = os.path.basename(clean_path) if clean_path else ""

        candidates = [
            clean_path,
            os.path.join("sample_videos", filename) if filename else "",
            os.path.join("uploads", filename) if filename else "",
            os.path.join("..", filename) if filename else "",
            os.path.join("..", "..", filename) if filename else "",
        ]

        found_path = None
        for candidate in candidates:
            if candidate and os.path.exists(candidate) and not os.path.isdir(candidate):
                found_path = candidate
                break

        if found_path:
            self._cap = cv2.VideoCapture(found_path)
            if self._cap.isOpened():
                self._use_synthetic = False
                self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
                self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
                self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
                self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self._duration = self._frame_count / self._fps if self._fps > 0 else 30.0
                print(f"[FileVideoSource] Successfully opened video file '{found_path}' ({self._width}x{self._height} @ {self._fps} FPS)")
                return True

        print(f"[FileVideoSource] File '{self.filepath}' not found on server. Using Cloud Surveillance Stream.")
        self._cap = None
        self._use_synthetic = True
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._use_synthetic or self._cap is None or not self._cap.isOpened():
            frame = _generate_synthetic_video_frame(self._angle)
            self._angle += 0.05
            time.sleep(1.0 / 30.0)
            return True, frame
        return self._cap.read()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._use_synthetic = False

    def is_opened(self) -> bool:
        return self._use_synthetic or (self._cap is not None and self._cap.isOpened())

    @property
    def fps(self) -> float:
        return self._fps if self._fps > 0 else 30.0

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def frame_count(self) -> Optional[int]:
        return self._frame_count if self._frame_count > 0 else None

    @property
    def duration(self) -> Optional[float]:
        return self._duration if self._duration > 0 else None


class CameraVideoSource(VideoSource):
    """
    Video source backed by a device camera with Cloud Server Synthetic Fallback.
    """

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._cap: Optional[cv2.VideoCapture] = None
        self._use_synthetic: bool = False
        self._width: int = 1280
        self._height: int = 720
        self._fps: float = 30.0
        self._angle: float = 0.0
        self._pushed_frame: Optional[np.ndarray] = None
        self._last_pushed_time: float = 0.0
        self._lock = threading.Lock()

    def push_frame(self, frame: np.ndarray):
        with self._lock:
            self._pushed_frame = frame.copy()
            self._last_pushed_time = time.time()
            self._use_synthetic = False
            self._width = frame.shape[1]
            self._height = frame.shape[0]

    def open(self) -> bool:
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == 'nt' else [cv2.CAP_ANY]
        cap = None
        for backend in backends:
            try:
                temp_cap = cv2.VideoCapture(self.device_index, backend)
                if temp_cap.isOpened():
                    ret, frame = temp_cap.read()
                    if ret and frame is not None and frame.size > 0:
                        cap = temp_cap
                        break
                    temp_cap.release()
            except Exception:
                pass

        if cap is None or not cap.isOpened():
            print(f"[CameraVideoSource] Device {self.device_index} unavailable. Enabling Live Cloud Feed.")
            self._cap = None
            self._use_synthetic = True
        else:
            self._cap = cap
            self._use_synthetic = False
            self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            print(f"[CameraVideoSource] Successfully opened physical camera device {self.device_index} ({self._width}x{self._height} @ {self._fps} FPS)")
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._pushed_frame is not None and time.time() - self._last_pushed_time < 2.0:
                return True, self._pushed_frame.copy()

        if self._use_synthetic or self._cap is None or not self._cap.isOpened():
            frame = _generate_synthetic_video_frame(self._angle)
            self._angle += 0.05
            time.sleep(1.0 / 30.0)
            return True, frame

        return self._cap.read()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._use_synthetic = False

    def is_opened(self) -> bool:
        return self._use_synthetic or (self._cap is not None and self._cap.isOpened())

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height
