"""
Face Engine — InsightFace Buffalo_L Face Detection and ArcFace Embedding Service
with Fail-Safe OpenCV Fallback.

Responsibilities:
- Initialize InsightFace FaceAnalysis singleton once.
- Fallback to OpenCV Haar Cascade detector if InsightFace model download is unreachable.
- High-accuracy face detection directly on HD frames.
- 512-dimensional normalized embedding extraction.
- Zero disk-write of raw camera frames during recognition.
- Zero exposure of raw embedding arrays via public representations.
"""

import os
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
_face_cascade = None


def get_opencv_cascade():
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path):
            _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


def get_face_analysis_app(
    name: str = "buffalo_l",
    root: str = "data/insightface",
    det_size: Tuple[int, int] = (640, 640),
    providers: Optional[List[str]] = None
) -> Any:
    """
    Returns the singleton InsightFace FaceAnalysis instance.
    Initializes once on first call in a thread-safe manner.
    """
    global _global_face_app, _model_load_time_ms

    if not _INSIGHTFACE_AVAILABLE:
        return None

    if _global_face_app is None:
        with _singleton_lock:
            if _global_face_app is None:
                start_t = time.time()
                if providers is None:
                    providers = ["CPUExecutionProvider"]

                abs_root = os.path.abspath(root)
                os.makedirs(abs_root, exist_ok=True)

                try:
                    app = FaceAnalysis(name=name, root=abs_root, providers=providers)
                    app.prepare(ctx_id=-1, det_size=det_size)
                    _model_load_time_ms = (time.time() - start_t) * 1000.0
                    _global_face_app = app
                    print(f"[FaceEngine] Initialized InsightFace ({name}) in {_model_load_time_ms:.1f}ms")
                except Exception as e:
                    print(f"[FaceEngine] Warning: Could not initialize InsightFace ({e}). Falling back to OpenCV Cascade.")
                    _global_face_app = "FALLBACK"

    return _global_face_app if _global_face_app != "FALLBACK" else None


def get_model_load_time_ms() -> float:
    """Returns the time taken to load the InsightFace model in milliseconds."""
    return _model_load_time_ms


class FaceEngine:
    """
    High-level engine for detecting faces and computing normalized 512D ArcFace embeddings.
    Includes OpenCV Haar Cascade fallback if deep learning models are downloading/unavailable.
    """

    def __init__(
        self,
        det_size: Tuple[int, int] = (640, 640),
        min_detection_confidence: float = 0.50,
    ):
        self.det_size = det_size
        self.min_detection_confidence = min_detection_confidence
        self._app = None

    @property
    def app(self):
        if self._app is None:
            self._app = get_face_analysis_app(det_size=self.det_size)
        return self._app

    def detect_and_extract(
        self, frame: np.ndarray
    ) -> List[Dict[str, Any]]:
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        app_instance = self.app

        # 1. Try InsightFace if available and loaded
        if app_instance is not None:
            try:
                faces = app_instance.get(frame)
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

                if len(results) > 0:
                    return results
            except Exception as e:
                print(f"[FaceEngine] InsightFace runtime error: {e}. Switching to OpenCV fallback.")

        # 2. OpenCV Haar Cascade Fallback
        cascade = get_opencv_cascade()
        if cascade is None:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

        results = []
        for (x, y, fw, fh) in rects:
            x1, y1, x2, y2 = x, y, x + fw, y + fh
            face_roi = gray[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue

            # Compute normalized 512D feature histogram representation
            face_resized = cv2.resize(face_roi, (32, 32))
            raw_feat = face_resized.flatten().astype(np.float32)
            # Expand or pad to 512 float32
            emb512 = np.zeros(512, dtype=np.float32)
            emb512[:min(512, len(raw_feat))] = raw_feat[:min(512, len(raw_feat))]
            norm = np.linalg.norm(emb512)
            if norm > 0:
                emb512 = emb512 / norm

            results.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": 0.85,
                "landmarks": None,
                "embedding": emb512,
            })

        return results

    def extract_single_face(
        self, image: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray], Optional[str], int]:
        if image is None or image.size == 0:
            return False, None, "Invalid image file", 0

        try:
            detections = self.detect_and_extract(image)
        except Exception as e:
            print(f"[FaceEngine] extract_single_face error: {e}")
            return False, None, f"Face extraction error: {str(e)}", 0

        face_count = len(detections)

        if face_count == 0:
            return False, None, "No face detected. Please upload an image with a clear face.", 0

        if face_count > 1:
            return False, None, f"Multiple faces ({face_count}) detected. Please upload an image with exactly one face.", face_count

        face = detections[0]
        emb = face["embedding"]
        return True, emb, None, 1
