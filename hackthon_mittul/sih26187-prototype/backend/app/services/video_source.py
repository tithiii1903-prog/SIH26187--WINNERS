"""
Video Source Abstraction Layer.

Provides a clean interface for reading video frames from different sources.
Currently implements FileVideoSource for MP4 files.
Designed for future extensibility (e.g., RTSPVideoSource).
"""

import cv2
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class VideoSource(ABC):
    """
    Abstract base class for video sources.
    Any video source (file, RTSP, USB camera) must implement this interface.
    The AI pipeline only depends on this interface — not on the source type.
    """

    @abstractmethod
    def open(self) -> bool:
        """Open the video source. Returns True if successful."""
        pass

    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame.
        Returns (success: bool, frame: np.ndarray or None).
        Frame is in BGR format (OpenCV default).
        """
        pass

    @abstractmethod
    def get_fps(self) -> float:
        """Return the source frame rate."""
        pass

    @abstractmethod
    def get_resolution(self) -> Tuple[int, int]:
        """Return (width, height) of the video source."""
        pass

    @abstractmethod
    def get_duration(self) -> Optional[float]:
        """Return duration in seconds, or None if not available (e.g., live stream)."""
        pass

    @abstractmethod
    def get_frame_count(self) -> Optional[int]:
        """Return total frame count, or None if not available."""
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """Return True if the source is currently open."""
        pass

    @abstractmethod
    def release(self):
        """Release all resources associated with this source."""
        pass

    def is_live(self) -> bool:
        """Return True if this is a live source (camera). Default: False (file)."""
        return False


class FileVideoSource(VideoSource):
    """
    Video source backed by a local MP4 file.
    Reads frames sequentially — never loads the entire video into RAM.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._cap: Optional[cv2.VideoCapture] = None
        self._width: int = 0
        self._height: int = 0
        self._fps: float = 0.0
        self._frame_count: int = 0
        self._duration: float = 0.0

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self.filepath)
        if not self._cap.isOpened():
            self._cap = None
            return False

        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._duration = self._frame_count / self._fps if self._fps > 0 else 0.0
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None
        ret, frame = self._cap.read()
        if not ret:
            return False, None
        return True, frame

    def get_fps(self) -> float:
        return self._fps

    def get_resolution(self) -> Tuple[int, int]:
        return self._width, self._height

    def get_duration(self) -> Optional[float]:
        return self._duration if self._duration > 0 else None

    def get_frame_count(self) -> Optional[int]:
        return self._frame_count if self._frame_count > 0 else None

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class CameraVideoSource(VideoSource):
    """
    Video source backed by a local device camera (USB, built-in, etc.).
    Opens via cv2.VideoCapture(device_index).

    Frame read failures are tolerated briefly (bounded retries) to handle
    transient USB glitches. If the camera truly disconnects, read_frame()
    returns (False, None) so the FrameProcessor can stop cleanly.
    """

    MAX_READ_RETRIES = 5
    RETRY_DELAY_SEC = 0.1

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._cap: Optional[cv2.VideoCapture] = None
        self._width: int = 0
        self._height: int = 0
        self._fps: float = 0.0

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self.device_index)
        if not self._cap.isOpened():
            self._cap = None
            return False

        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None

        # First attempt
        ret, frame = self._cap.read()
        if ret:
            return True, frame

        # Bounded retries for transient failures (USB glitch, etc.)
        import time
        for _ in range(self.MAX_READ_RETRIES):
            time.sleep(self.RETRY_DELAY_SEC)
            ret, frame = self._cap.read()
            if ret:
                return True, frame

        # All retries exhausted — camera is disconnected or unavailable
        return False, None

    def get_fps(self) -> float:
        return self._fps

    def get_resolution(self) -> Tuple[int, int]:
        return self._width, self._height

    def get_duration(self) -> Optional[float]:
        return None  # Live source — no duration

    def get_frame_count(self) -> Optional[int]:
        return None  # Live source — no frame count

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def is_live(self) -> bool:
        return True
