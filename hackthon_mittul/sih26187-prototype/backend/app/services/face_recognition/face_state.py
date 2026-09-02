"""
Face State Tracker — Independent Per-Face State, Temporal Stability, and Lifecycle Isolation.

Responsibilities:
- Maintain independent state for each detected face without global contamination.
- Associate face detections across consecutive frames via spatial bounding box overlap (IoU).
- Enforce temporal stability: require >= 2 consecutive matches before confirming match status.
- Provide graceful disappearance handling (short grace period before pruning stale faces).
- Ensure match on Face A never leaks to Face B or Face C.
- Cleanly clear state when a face disappears while preserving other faces.
"""

import time
import threading
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


def compute_bbox_iou(box1: List[int], box2: List[int]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])

    inter_w = max(0, x2_min - x1_max)
    inter_h = max(0, y2_min - y1_max)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


class FaceStateTracker:
    """
    Tracks face recognition state across consecutive frames with spatial isolation.
    """

    def __init__(
        self,
        min_iou_match: float = 0.30,
        min_consecutive_matches: int = 2,
        grace_period_sec: float = 0.60,
    ):
        self.min_iou_match = min_iou_match
        self.min_consecutive_matches = min_consecutive_matches
        self.grace_period_sec = grace_period_sec

        self._lock = threading.Lock()
        self._next_face_id: int = 1
        self._matched_emitted: set = set()  # set of face_ids that already emitted match event

        # face_id -> {
        #     "face_id": int,
        #     "bbox": [x1, y1, x2, y2],
        #     "confidence": float,
        #     "candidate_id": Optional[str],
        #     "candidate_name": Optional[str],
        #     "candidate_status": Optional[str],
        #     "candidate_similarity": float,
        #     "consecutive_matches": int,
        #     "matched": bool,
        #     "person_id": Optional[str],
        #     "name": Optional[str],
        #     "status": Optional[str],
        #     "similarity": float,
        #     "first_seen": float,
        #     "last_seen": float,
        # }
        self._active_states: Dict[int, Dict[str, Any]] = {}

    def reset(self):
        """Clears all tracked face states."""
        with self._lock:
            self._active_states.clear()
            self._matched_emitted.clear()
            self._next_face_id = 1

    def update(
        self,
        detections: List[Dict[str, Any]],
        matcher_func,
        current_time: Optional[float] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Updates face state given current frame detections and a matcher function.

        Args:
            detections: list of dicts from FaceEngine with 'bbox', 'confidence', 'embedding'.
            matcher_func: callable taking embedding -> dict {"matched", "person_id", "name", "status", "similarity"}.
            current_time: optional timestamp (defaults to time.time()).

        Returns:
            Tuple of (visible_faces, generated_events)
        """
        if current_time is None:
            current_time = time.time()

        generated_events: List[Dict[str, Any]] = []

        with self._lock:
            existing_ids = list(self._active_states.keys())
            matched_detection_indices = set()
            matched_state_ids = set()

            # 1. Greedy bipartite matching based on IoU overlap
            iou_matrix = []
            for det_idx, det in enumerate(detections):
                for sid in existing_ids:
                    st_box = self._active_states[sid]["bbox"]
                    iou = compute_bbox_iou(det["bbox"], st_box)
                    if iou >= self.min_iou_match:
                        iou_matrix.append((iou, det_idx, sid))

            # Sort by IoU descending
            iou_matrix.sort(key=lambda item: item[0], reverse=True)

            for iou, det_idx, sid in iou_matrix:
                if det_idx in matched_detection_indices or sid in matched_state_ids:
                    continue
                matched_detection_indices.add(det_idx)
                matched_state_ids.add(sid)

                # Update existing face state
                det = detections[det_idx]
                st = self._active_states[sid]
                st["bbox"] = det["bbox"]
                st["confidence"] = det["confidence"]
                st["last_seen"] = current_time

                # Match against registered faces
                match_res = matcher_func(det["embedding"])

                if match_res.get("matched", False):
                    cand_id = match_res.get("person_id")
                    if cand_id == st["candidate_id"]:
                        st["consecutive_matches"] += 1
                    else:
                        st["candidate_id"] = cand_id
                        st["candidate_name"] = match_res.get("name")
                        st["candidate_status"] = match_res.get("status")
                        st["consecutive_matches"] = 1

                    st["candidate_similarity"] = match_res.get("similarity", 0.0)

                    # Temporal confirmation: require >= min_consecutive_matches
                    if st["consecutive_matches"] >= self.min_consecutive_matches:
                        was_matched = st["matched"]
                        st["matched"] = True
                        st["person_id"] = st["candidate_id"]
                        st["name"] = st["candidate_name"]
                        st["status"] = st["candidate_status"]
                        st["similarity"] = st["candidate_similarity"]

                        # Emit match event once per tracked face session
                        if not was_matched or sid not in self._matched_emitted:
                            self._matched_emitted.add(sid)
                            generated_events.append({
                                "event_type": "FACE_WATCHLIST_MATCH",
                                "face_id": sid,
                                "watchlist_id": st["person_id"],
                                "name": st["name"],
                                "status": st["status"],
                                "similarity": st["similarity"],
                                "timestamp": current_time,
                            })
                    else:
                        if not st["matched"]:
                            st["similarity"] = st["candidate_similarity"]
                else:
                    # Live embedding did not match
                    if not st["matched"]:
                        st["candidate_id"] = None
                        st["candidate_name"] = None
                        st["candidate_status"] = None
                        st["consecutive_matches"] = 0
                        st["similarity"] = match_res.get("similarity", 0.0)

            # 2. Allocate new states for unmatched detections
            for det_idx, det in enumerate(detections):
                if det_idx in matched_detection_indices:
                    continue

                face_id = self._next_face_id
                self._next_face_id += 1

                match_res = matcher_func(det["embedding"])
                matched_now = False
                cand_id = None
                cand_name = None
                cand_status = None
                sim = match_res.get("similarity", 0.0)
                consecutive = 0

                if match_res.get("matched", False):
                    cand_id = match_res.get("person_id")
                    cand_name = match_res.get("name")
                    cand_status = match_res.get("status")
                    consecutive = 1
                    if self.min_consecutive_matches <= 1:
                        matched_now = True
                        self._matched_emitted.add(face_id)
                        generated_events.append({
                            "event_type": "FACE_WATCHLIST_MATCH",
                            "face_id": face_id,
                            "watchlist_id": cand_id,
                            "name": cand_name,
                            "status": cand_status,
                            "similarity": sim,
                            "timestamp": current_time,
                        })

                self._active_states[face_id] = {
                    "face_id": face_id,
                    "bbox": det["bbox"],
                    "confidence": det["confidence"],
                    "candidate_id": cand_id,
                    "candidate_name": cand_name,
                    "candidate_status": cand_status,
                    "candidate_similarity": sim,
                    "consecutive_matches": consecutive,
                    "matched": matched_now,
                    "person_id": cand_id if matched_now else None,
                    "name": cand_name if matched_now else None,
                    "status": cand_status if matched_now else None,
                    "similarity": sim,
                    "first_seen": current_time,
                    "last_seen": current_time,
                }
                matched_state_ids.add(face_id)

            # 3. Handle disappearance & grace period
            stale_ids = []
            for sid, st in list(self._active_states.items()):
                if sid not in matched_state_ids:
                    time_since_seen = current_time - st["last_seen"]
                    if time_since_seen > self.grace_period_sec:
                        stale_ids.append(sid)
                        if st["matched"]:
                            generated_events.append({
                                "event_type": "FACE_WATCHLIST_MATCH_CLEARED",
                                "face_id": sid,
                                "watchlist_id": st["person_id"],
                                "name": st["name"],
                                "status": st["status"],
                                "similarity": st["similarity"],
                                "timestamp": current_time,
                            })

            for sid in stale_ids:
                self._matched_emitted.discard(sid)
                del self._active_states[sid]

            # 4. Construct current visible face outputs (only for faces visible or in grace)
            output_faces = []
            for sid in matched_state_ids:
                if sid in self._active_states:
                    st = self._active_states[sid]
                    output_faces.append({
                        "face_id": st["face_id"],
                        "bbox": list(st["bbox"]),
                        "name": st["name"] if st["matched"] else None,
                        "status": st["status"] if st["matched"] else None,
                        "similarity": float(st["similarity"]),
                        "matched": bool(st["matched"]),
                        "confidence": float(st["confidence"]),
                    })

            return output_faces, generated_events

    def get_active_states(self) -> Dict[int, Dict[str, Any]]:
        with self._lock:
            return {k: v.copy() for k, v in self._active_states.items()}
