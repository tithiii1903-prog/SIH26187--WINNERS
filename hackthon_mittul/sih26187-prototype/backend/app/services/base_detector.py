import torch
import gc
from ultralytics import YOLO

# Enforce single-thread CPU execution and disable gradient tracking to minimize memory
try:
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
except Exception:
    pass

_model_cache = {}

def get_yolo_model(model_path="yolov8n.pt"):
    """
    Returns a cached YOLO model instance and the optimal device (mps or cpu).
    """
    if model_path not in _model_cache:
        _model_cache[model_path] = YOLO(model_path)
        gc.collect()
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
