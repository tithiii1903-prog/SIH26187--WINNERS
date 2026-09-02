import json
import cv2
import numpy as np
from typing import List, Dict, Any
from .. import database

class VirtualFence:
    def __init__(self):
        self.zones = self._load_config()
        self.active_intrusions = {} # track_id -> zone_id

    def reload_zones(self):
        """Hot-reload zone configuration from DB. Used by live pipeline."""
        self.zones = self._load_config()
        # Clear active intrusions to re-evaluate with new zones
        self.active_intrusions = {}

    def reset(self):
        """Clears active intrusion state when a feed/session is restarted."""
        self.active_intrusions.clear()

    def _load_config(self) -> List[Dict[str, Any]]:
        try:
            return database.get_all_zones()
        except Exception as e:
            print(f"Failed to load zone config from DB: {e}")
            return []

    def process_frame(self, detections: List[Dict[str, Any]], timestamp: float) -> List[Dict[str, Any]]:
        events = []
        current_frame_intrusions = set() # (track_id, zone_id)

        for det in detections:
            track_id = det.get("id")
            if track_id is None:
                continue

            # Bounding box center
            box = det["box"]
            cx = (box[0] + box[2]) // 2
            cy = (box[1] + box[3]) // 2
            point = (cx, cy)

            for zone in self.zones:
                if not zone.get("enabled", True):
                    continue
                    
                zone_id = zone["id"]
                zone_name = zone["name"]
                polygon = np.array(zone["polygon"], np.int32)
                if len(polygon) < 3:
                    continue

                # Check point in polygon
                inside = cv2.pointPolygonTest(polygon, point, False) >= 0
                
                if inside:
                    current_frame_intrusions.add((track_id, zone_id))
                    
                    if track_id not in self.active_intrusions or self.active_intrusions[track_id] != zone_id:
                        # INTRUSION ENTER EVENT
                        self.active_intrusions[track_id] = zone_id
                        desc = f"Track ID {track_id} entered {zone_name}"
                        # Generate DB event asynchronously-ish or direct
                        try:
                            database.insert_event(
                                event_type="INTRUSION_ENTER",
                                timestamp=timestamp,
                                message=desc,
                                track_id=track_id,
                                zone_id=zone_id
                            )
                        except Exception as e:
                            print(f"Failed to log INTRUSION_ENTER event: {e}")
                            
                        events.append({
                            "timestamp": round(timestamp, 2),
                            "type": "INTRUSION_ENTER",
                            "zone_id": zone_id,
                            "zone_name": zone_name,
                            "track_id": track_id,
                            "description": desc
                        })

        # Check for exits
        exited_intrusions = []
        for track_id, zone_id in self.active_intrusions.items():
            if (track_id, zone_id) not in current_frame_intrusions:
                exited_intrusions.append(track_id)
                # Find zone name
                zone_name = next((z["name"] for z in self.zones if z["id"] == zone_id), zone_id)
                desc = f"Track ID {track_id} exited {zone_name}"
                
                try:
                    database.insert_event(
                        event_type="INTRUSION_EXIT",
                        timestamp=timestamp,
                        message=desc,
                        track_id=track_id,
                        zone_id=zone_id
                    )
                except Exception as e:
                    print(f"Failed to log INTRUSION_EXIT event: {e}")
                    
                events.append({
                    "timestamp": round(timestamp, 2),
                    "type": "INTRUSION_EXIT",
                    "zone_id": zone_id,
                    "zone_name": zone_name,
                    "track_id": track_id,
                    "description": desc
                })

        for track_id in exited_intrusions:
            del self.active_intrusions[track_id]

        return events

    def get_zones(self):
        return self.zones

    def get_active_intrusions(self):
        return self.active_intrusions

