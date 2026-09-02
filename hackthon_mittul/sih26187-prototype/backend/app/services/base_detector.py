import torch
from ultralytics import YOLO

_model_cache = {}

def get_yolo_model(model_path="yolov8n.pt"):
    """
    Returns a cached YOLO model instance and the optimal device (mps or cpu).
    """
    if model_path not in _model_cache:
        _model_cache[model_path] = YOLO(model_path)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return _model_cache[model_path], device

class BaseYOLODetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Initializes the base YOLO detector.
        Uses cached model to reduce memory duplication if multiple detectors use the same weights.
        Automatically checks and uses Apple MPS if available, otherwise CPU.
        """
        self.model, self.device = get_yolo_model(model_path)
