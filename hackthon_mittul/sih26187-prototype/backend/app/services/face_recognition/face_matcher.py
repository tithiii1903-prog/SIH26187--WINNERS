"""
Face Matcher — Cosine Similarity Matching for ArcFace 512D Embeddings.

Responsibilities:
- Maintain registered face candidates in memory.
- Compare live normalized 512D embeddings against registered embeddings via dot product.
- Select the candidate with highest similarity.
- Apply security threshold (default 0.70).
- Return structured match output without exposing raw embedding arrays.
"""

import threading
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


DEFAULT_FACE_THRESHOLD = 0.70


class FaceMatcher:
    """
    Thread-safe face matcher using Cosine Similarity on normalized 512D ArcFace embeddings.
    """

    def __init__(self, match_threshold: float = DEFAULT_FACE_THRESHOLD):
        self.match_threshold = match_threshold
        self._lock = threading.Lock()
        # id -> {"id": str, "name": str, "status": str, "embedding": np.ndarray}
        self._registered_faces: Dict[str, Dict[str, Any]] = {}

    @property
    def threshold(self) -> float:
        return self.match_threshold

    @threshold.setter
    def threshold(self, value: float):
        self.match_threshold = max(0.0, min(1.0, float(value)))

    def clear_registered(self):
        """Clear all registered faces."""
        with self._lock:
            self._registered_faces.clear()

    def register_face(
        self,
        person_id: str,
        name: str,
        status: str,
        embedding: np.ndarray,
        enabled: bool = True
    ):
        """
        Registers or updates a known face record.
        Ensures embedding is L2-normalized.
        """
        if embedding is None:
            return

        norm = np.linalg.norm(embedding)
        if norm > 0:
            norm_emb = (embedding / norm).astype(np.float32)
        else:
            norm_emb = embedding.astype(np.float32)

        with self._lock:
            self._registered_faces[str(person_id)] = {
                "id": str(person_id),
                "name": str(name),
                "status": str(status).upper(),
                "embedding": norm_emb,
                "enabled": bool(enabled),
            }

    def unregister_face(self, person_id: str) -> bool:
        """Removes a registered face by ID."""
        with self._lock:
            return self._registered_faces.pop(str(person_id), None) is not None

    def set_registered_faces(self, records: List[Dict[str, Any]]):
        """
        Replaces all registered face records at once.
        Each record must contain: 'id', 'name', 'status', 'embedding' (and optional 'enabled').
        """
        new_map: Dict[str, Dict[str, Any]] = {}
        for r in records:
            pid = str(r["id"])
            emb = r.get("embedding")
            if emb is None:
                continue
            norm = np.linalg.norm(emb)
            if norm > 0:
                norm_emb = (emb / norm).astype(np.float32)
            else:
                norm_emb = emb.astype(np.float32)

            new_map[pid] = {
                "id": pid,
                "name": str(r.get("name", "Unknown")),
                "status": str(r.get("status", "WATCHLIST")).upper(),
                "embedding": norm_emb,
                "enabled": bool(r.get("enabled", True)),
            }

        with self._lock:
            self._registered_faces = new_map

    def get_registered_count(self) -> int:
        with self._lock:
            return len([f for f in self._registered_faces.values() if f.get("enabled", True)])

    def match(
        self,
        live_embedding: np.ndarray,
        custom_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Compares live 512D face embedding against all active registered face embeddings.
        Cosine similarity = dot product of L2-normalized vectors.

        Returns:
            {
                "matched": bool,
                "person_id": Optional[str],
                "name": Optional[str],
                "status": Optional[str],
                "similarity": float
            }
        """
        threshold = custom_threshold if custom_threshold is not None else self.match_threshold

        if live_embedding is None or live_embedding.size == 0:
            return {
                "matched": False,
                "person_id": None,
                "name": None,
                "status": None,
                "similarity": 0.0,
            }

        norm = np.linalg.norm(live_embedding)
        if norm > 0:
            live_norm = (live_embedding / norm).astype(np.float32)
        else:
            return {
                "matched": False,
                "person_id": None,
                "name": None,
                "status": None,
                "similarity": 0.0,
            }

        with self._lock:
            candidates = [
                rec for rec in self._registered_faces.values()
                if rec.get("enabled", True)
            ]

        if not candidates:
            return {
                "matched": False,
                "person_id": None,
                "name": None,
                "status": None,
                "similarity": 0.0,
            }

        best_match: Optional[Dict[str, Any]] = None
        highest_similarity: float = -1.0

        for candidate in candidates:
            cand_emb = candidate["embedding"]
            similarity = float(np.dot(live_norm, cand_emb))
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = candidate

        highest_similarity_clamped = max(0.0, min(1.0, highest_similarity))

        if highest_similarity >= threshold and best_match is not None:
            return {
                "matched": True,
                "person_id": best_match["id"],
                "name": best_match["name"],
                "status": best_match["status"],
                "similarity": round(highest_similarity_clamped, 4),
            }

        return {
            "matched": False,
            "person_id": None,
            "name": None,
            "status": None,
            "similarity": round(highest_similarity_clamped, 4),
        }

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Utility to calculate cosine similarity directly between two embeddings."""
        if emb1 is None or emb2 is None:
            return 0.0
        n1 = np.linalg.norm(emb1)
        n2 = np.linalg.norm(emb2)
        if n1 == 0 or n2 == 0:
            return 0.0
        v1 = emb1 / n1
        v2 = emb2 / n2
        sim = float(np.dot(v1, v2))
        return max(-1.0, min(1.0, sim))
