from typing import Dict, List, Tuple, Set, Optional, Any
import time


class PersonTracker:
    def __init__(
        self,
        max_history: int = 30,
        grace_period_sec: float = 0.8,
        max_inactive_frames: int = 15,
    ):
        """
        Manages track state and trajectories for detected persons with time-based
        inactivity grace logic to prevent flickering across variable AI frame rates.
        Does NOT perform bounding box tracking/association (handled by YOLO+ByteTrack).

        Args:
            max_history (int): Maximum number of center points to store per track ID.
            grace_period_sec (float): Time in seconds before an unseen track is declared disappeared (default 0.8s).
            max_inactive_frames (int): Minimum frames before expiry check fallback.
        """
        self.max_history = max_history
        self.grace_period_sec = grace_period_sec
        self.max_inactive_frames = max_inactive_frames

        # Per-track isolated state dictionaries
        self.trajectories: Dict[int, List[Tuple[int, int]]] = {}
        self.last_seen_time: Dict[int, float] = {}
        self.last_seen_frame: Dict[int, int] = {}
        self.last_known_box: Dict[int, List[int]] = {}
        self.last_known_conf: Dict[int, float] = {}

        # Set of currently confirmed active tracks
        self.active_tracks: Set[int] = set()

        # Tracking telemetry
        self.max_active_tracks: int = 0

    def reset(self):
        """Safely resets all tracking state and histories for a fresh feed session."""
        self.trajectories.clear()
        self.last_seen_time.clear()
        self.last_seen_frame.clear()
        self.last_known_box.clear()
        self.last_known_conf.clear()
        self.active_tracks.clear()
        self.max_active_tracks = 0

    def update_trajectories(
        self,
        detections: List[Dict[str, Any]],
        frame_idx: int,
        timestamp: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], List[int], List[int]]:
        """
        Updates the trajectory history based on the latest tracked detections.
        Applies time-based grace period to avoid premature track deletion on transient misses.

        Args:
            detections (List[Dict]): List of detection dictionaries (each contains 'id', 'box', 'conf').
            frame_idx (int): The current frame index.
            timestamp (Optional[float]): Monotonic or frame timestamp. Uses time.time() if None.

        Returns:
            Tuple[List[Dict], List[int], List[int]]:
                - augmented detections (each with 'trajectory')
                - new_track_ids (confirmed new track IDs)
                - disappeared_track_ids (track IDs expired after grace period)
        """
        now = time.time() if timestamp is None else timestamp
        new_track_ids: List[int] = []
        current_seen_ids: Set[int] = set()

        for det in detections:
            track_id = det.get("id")
            if track_id is None:
                continue

            current_seen_ids.add(track_id)

            if track_id not in self.active_tracks:
                self.active_tracks.add(track_id)
                new_track_ids.append(track_id)

            box = det["box"]
            center_x = (box[0] + box[2]) // 2
            center_y = (box[1] + box[3]) // 2
            center = (center_x, center_y)

            # Update trajectory
            if track_id not in self.trajectories:
                self.trajectories[track_id] = []

            self.trajectories[track_id].append(center)

            # Limit history
            if len(self.trajectories[track_id]) > self.max_history:
                self.trajectories[track_id] = self.trajectories[track_id][-self.max_history:]

            self.last_seen_time[track_id] = now
            self.last_seen_frame[track_id] = frame_idx
            self.last_known_box[track_id] = list(box)
            self.last_known_conf[track_id] = float(det.get("conf", 0.0))

            # Attach trajectory to detection dict
            det["trajectory"] = self.trajectories[track_id].copy()

        # Check for tracks that have genuinely expired (exceeded grace period)
        disappeared_track_ids: List[int] = []
        stale_ids: List[int] = []

        for track_id, last_t in list(self.last_seen_time.items()):
            if track_id in current_seen_ids:
                continue
            time_since_seen = now - last_t
            if time_since_seen > self.grace_period_sec:
                stale_ids.append(track_id)

        for track_id in stale_ids:
            disappeared_track_ids.append(track_id)
            self.active_tracks.discard(track_id)
            self.trajectories.pop(track_id, None)
            self.last_seen_time.pop(track_id, None)
            self.last_seen_frame.pop(track_id, None)
            self.last_known_box.pop(track_id, None)
            self.last_known_conf.pop(track_id, None)

        # Update peak telemetry
        if len(self.active_tracks) > self.max_active_tracks:
            self.max_active_tracks = len(self.active_tracks)

        return detections, new_track_ids, disappeared_track_ids

    def get_active_tracks_snapshot(self, now: Optional[float] = None) -> Dict[int, Dict[str, Any]]:
        """
        Returns active tracks that are still valid within the grace period.
        Used for smoothing over transient missed AI frames.
        """
        current_time = time.time() if now is None else now
        snapshot: Dict[int, Dict[str, Any]] = {}
        for track_id in list(self.active_tracks):
            last_t = self.last_seen_time.get(track_id, 0.0)
            if current_time - last_t <= self.grace_period_sec:
                snapshot[track_id] = {
                    "id": track_id,
                    "box": self.last_known_box.get(track_id, [0, 0, 0, 0]),
                    "conf": self.last_known_conf.get(track_id, 0.0),
                    "trajectory": self.trajectories.get(track_id, []).copy(),
                    "last_seen_time": last_t,
                    "last_seen_frame": self.last_seen_frame.get(track_id, 0),
                }
        return snapshot
