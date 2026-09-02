"""
Isolated HD Face Recognition Module (InsightFace Buffalo_L + ArcFace 512D).

This module is completely decoupled from FrameProcessor, YOLO PersonDetector,
ByteTrack PersonTracker, VehicleDetector, and VirtualFence.
"""

from .face_engine import FaceEngine, get_face_analysis_app, get_model_load_time_ms
from .face_matcher import FaceMatcher, DEFAULT_FACE_THRESHOLD
from .face_state import FaceStateTracker, compute_bbox_iou
from .face_camera import FaceCamera
from .watchlist_service import WatchlistService

__all__ = [
    "FaceEngine",
    "get_face_analysis_app",
    "get_model_load_time_ms",
    "FaceMatcher",
    "DEFAULT_FACE_THRESHOLD",
    "FaceStateTracker",
    "compute_bbox_iou",
    "FaceCamera",
    "WatchlistService",
]
