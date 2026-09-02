"""
Face Engine — InsightFace Buffalo_L Face Detection and ArcFace Embedding Service.

Responsibilities:
- Initialize InsightFace FaceAnalysis singleton once.
- Reuse singleton model across frames and threads.
- High-accuracy face detection directly on HD frames (RetinaFace det_10g).
- 512-dimensional ArcFace embedding extraction (w600k_r50).
- L2-normalization of face embeddings.
- Face landmark extraction (5-point keypoints / 3D 68 / 2D 106).
- Zero disk-write of raw camera frames during recognition.
- Zero exposure of raw embedding arrays via public representations.
"""

import time
import threading
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2

try:
    import insightface
    from insightface.app import FaceAnalysis
    _INSIGHTFACE_AVAILABLE = True
except ImportError:
    _INSIGHTFACE_AVAILABLE = False
    FaceAnalysis = None


_singleton_lock = threading.Lock()
_global_face_app: Optional[Any] = None
_model_load_time_ms: float = 0.0


def get_face_analysis_app(
    name: str = "buffalo_l",
    root: str = "~/.insightface",
    det_size: Tuple[int, int] = (640, 640),
    providers: Optional[List[str]] = None
) -> Any:
    """
    Returns the singleton InsightFace FaceAnalysis instance.
    Initializes once on first call in a thread-safe manner.
    """
    global _global_face_app, _model_load_time_ms

    if not _INSIGHTFACE_AVAILABLE:
        raise RuntimeError(
            "insightface package is not installed. Please install onnxruntime and insightface."
        )

    if _global_face_app is None:
        with _singleton_lock:
            if _global_face_app is None:
                start_t = time.time()
                if providers is None:
                    providers = ["CPUExecutionProvider"]

                app = FaceAnalysis(name=name, root=root, providers=providers)
                app.prepare(ctx_id=-1, det_size=det_size)
                _model_load_time_ms = (time.time() - start_t) * 1000.0
                _global_face_app = app
                print(f"[FaceEngine] Initialized InsightFace ({name}) in {_model_load_time_ms:.1f}ms")

    return _global_face_app


def get_model_load_time_ms() -> float:
    """Returns the time taken to load the InsightFace model in milliseconds."""
    return _model_load_time_ms


class FaceEngine:
    """
    High-level engine for detecting faces and computing normalized 512D ArcFace embeddings.
    """

    def __init__(
        self,
        det_size: Tuple[int, int] = (640, 640),
        min_detection_confidence: float = 0.50,
    ):
        self.det_size = det_size
        self.min_detection_confidence = min_detection_confidence
        self._app = get_face_analysis_app(det_size=det_size)

    def detect_and_extract(
        self, frame: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Detects all faces directly in the given HD frame and extracts L2-normalized embeddings.

        Args:
            frame: BGR numpy image frame (H x W x 3).

        Returns:
            List of face detection dicts:
            [
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float,
                    "landmarks": list of [x, y] or None,
                    "embedding": np.ndarray (512,),  # L2 normalized float32
                },
                ...
            ]
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        faces = self._app.get(frame)

        results = []
        for face in faces:
            det_score = float(getattr(face, "det_score", 0.0))
            if det_score < self.min_detection_confidence:
                continue

            raw_bbox = face.bbox
            x1 = max(0, int(round(raw_bbox[0])))
            y1 = max(0, int(round(raw_bbox[1])))
            x2 = min(w, int(round(raw_bbox[2])))
            y2 = min(h, int(round(raw_bbox[3])))

            # Skip degenerate boxes
            if x2 <= x1 or y2 <= y1:
                continue

            raw_emb = face.embedding
            if raw_emb is None:
                continue

            norm = np.linalg.norm(raw_emb)
            if norm > 0:
                norm_emb = (raw_emb / norm).astype(np.float32)
            else:
                norm_emb = np.zeros(512, dtype=np.float32)

            landmarks = None
            if hasattr(face, "kps") and face.kps is not None:
                landmarks = face.kps.tolist()

            results.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": det_score,
                "landmarks": landmarks,
                "embedding": norm_emb,
            })

        return results

    def extract_single_face(
        self, image: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray], Optional[str], int]:
        """
        Validates an enrollment image for exactly one high-quality face and returns
        the normalized 512D ArcFace embedding.

        Returns:
            (success: bool, embedding: np.ndarray or None, error_message: str or None, face_count: int)
        """
        if image is None or image.size == 0:
            return False, None, "Invalid image", 0

        detections = self.detect_and_extract(image)
        face_count = len(detections)

        if face_count == 0:
            return False, None, "No face detected. Please ensure your face is clearly visible.", 0

        if face_count > 1:
            return False, None, f"Multiple faces ({face_count}) detected. Please upload an image with exactly one face.", face_count

        face = detections[0]
        emb = face["embedding"]
        return True, emb, None, 1
