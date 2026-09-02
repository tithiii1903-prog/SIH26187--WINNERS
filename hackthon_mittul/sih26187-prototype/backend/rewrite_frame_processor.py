import re

with open("/Users/arshmeen/hackthon_mittul/sih26187-prototype/backend/app/services/frame_processor.py", "r") as f:
    content = f.read()

# 1. Update __init__ analytics dictionary to include stream_fps
content = content.replace('"processing_fps": 0.0,', '"processing_fps": 0.0,\n            "stream_fps": 0.0,')

# 2. Add TARGET_AI_FPS and caching variables
state_replace = """        # State
        self._status = "READY"  # READY, STARTING, LIVE, STOPPING, STOPPED, ERROR
        self._error_message: Optional[str] = None

        self.TARGET_AI_FPS = 10.0
        self._latest_person_detections = []
        self._latest_vehicle_detections = []
        self._latest_watchlist_matches = {}
        self._latest_face_boxes = {}
        self._latest_active_intrusions = {}
        self._latest_active_tracks_info = []
        self._ai_fps = 0.0
        self._stream_fps = 0.0
"""
content = re.sub(r'        # State.*?        self._error_message: Optional\[str\] = None\n', state_replace, content, flags=re.DOTALL)

# 3. Modify _processing_loop to decouple AI
loop_pattern = r'            frames_processed = 0\n            source_read_failed = False\n\n            while not self\._stop_event\.is_set\(\):\n(.*?)(?=\n                # 6\. Overlay HUD)'

new_loop_logic = """            frames_processed = 0
            source_read_failed = False
            last_ai_time = 0.0
            ai_interval = 1.0 / self.TARGET_AI_FPS

            while not self._stop_event.is_set():
                frame_start = time.time()

                # Check if fence reload is requested
                with self._lock:
                    if self._fence_reload_requested:
                        self._fence_reload_requested = False
                        self.virtual_fence.reload_zones(self.fence_config_path)

                # Read next frame
                success, frame = self.video_source.read_frame()
                if not success:
                    source_read_failed = True
                    break

                timestamp = frames_processed / source_fps if source_fps > 0 else 0.0

                # Get current module states
                with self._lock:
                    modules = self._modules_enabled.copy()

                annotated_frame = frame.copy()
                new_events = []
                current_time = time.time()

                # --- AI Processing Cadence ---
                if current_time - last_ai_time >= ai_interval:
                    last_ai_time = current_time
                    ai_start = time.time()

                    # 1. Person Detection + Tracking
                    if modules["human_detection"]:
                        if modules["human_tracking"]:
                            self._latest_person_detections = self.person_detector.track(frame)
                            self._latest_person_detections = self.person_tracker.update_trajectories(
                                self._latest_person_detections, frames_processed
                            )
                        else:
                            self._latest_person_detections = self.person_detector.detect(frame)
                    else:
                        self._latest_person_detections = []

                    # 2. Track events
                    current_ids: Set[int] = set()
                    self._latest_active_tracks_info = []
                    for det in self._latest_person_detections:
                        track_id = det.get("id")
                        if track_id is not None:
                            current_ids.add(track_id)
                            self._latest_active_tracks_info.append({
                                "id": track_id,
                                "conf": round(float(det["conf"]), 2),
                            })

                    if modules["human_detection"] and modules["human_tracking"]:
                        for tid in current_ids - self._active_ids_prev:
                            new_events.append({
                                "timestamp": round(timestamp, 2),
                                "type": "NEW_TRACK",
                                "track_id": tid,
                                "description": f"Track ID {tid} entered the frame.",
                            })
                        for tid in self._active_ids_prev - current_ids:
                            new_events.append({
                                "timestamp": round(timestamp, 2),
                                "type": "TRACK_DISAPPEARED",
                                "track_id": tid,
                                "description": f"Track ID {tid} left the frame.",
                            })
                    self._active_ids_prev = current_ids

                    # 2.5. Face Watchlist Matching
                    if (
                        modules["face_watchlist"]
                        and modules["human_detection"]
                        and self.face_watchlist is not None
                    ):
                        self._latest_watchlist_matches, wl_events, self._latest_face_boxes = self.face_watchlist.match_faces(
                            frame, self._latest_person_detections, frames_processed
                        )
                        new_events.extend(wl_events)
                    else:
                        self._latest_watchlist_matches = {}
                        self._latest_face_boxes = {}

                    # 3. Virtual Fence
                    if modules["virtual_fence"] and modules["human_detection"]:
                        fence_events = self.virtual_fence.process_frame(
                            self._latest_person_detections, timestamp
                        )
                        new_events.extend(fence_events)
                        self._latest_active_intrusions = self.virtual_fence.get_active_intrusions()
                    else:
                        self._latest_active_intrusions = {}

                    # 5. Vehicle Detection
                    if modules["vehicle_detection"]:
                        self._latest_vehicle_detections = self.vehicle_detector.detect(frame)
                    else:
                        self._latest_vehicle_detections = []

                    ai_time = time.time() - ai_start
                    self._ai_fps = 1.0 / ai_time if ai_time > 0 else 0.0

                # --- Draw Latest States on Current Frame ---
                
                if modules["virtual_fence"] and modules["human_detection"]:
                    # Draw fence zones
                    for zone in self.virtual_fence.get_zones():
                        if not zone.get("enabled", True):
                            continue
                        polygon = np.array(zone["polygon"], np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [polygon], True, (0, 0, 255), 2)
                        cv2.putText(
                            annotated_frame, zone["name"],
                            tuple(zone["polygon"][0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                        )

                # 4. Annotate persons
                if modules["human_detection"]:
                    for det in self._latest_person_detections:
                        box = det["box"]
                        conf = det["conf"]
                        track_id = det.get("id")

                        # Check for watchlist match
                        wl_match = (
                            self._latest_watchlist_matches.get(track_id)
                            if track_id is not None
                            else None
                        )

                        is_intruding = (
                            track_id is not None
                            and track_id in self._latest_active_intrusions
                            and modules["virtual_fence"]
                        )

                        # Watchlist match = RED, intrusion = RED, normal = GREEN
                        if wl_match:
                            box_color = (0, 0, 255)  # RED for watchlist
                        elif is_intruding:
                            box_color = (0, 0, 255)
                        else:
                            box_color = (0, 255, 0)

                        thickness = 3 if wl_match else 2
                        cv2.rectangle(
                            annotated_frame,
                            (box[0], box[1]), (box[2], box[3]),
                            box_color, thickness,
                        )

                        if track_id is not None and modules["human_tracking"]:
                            label = f"Track ID: {track_id} ({conf:.2f})"
                        else:
                            label = f"Person {conf:.2f}"
                        if is_intruding:
                            label += " - INTRUSION"

                        (lw, lh), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                        )
                        cv2.rectangle(
                            annotated_frame,
                            (box[0], box[1] - lh - baseline),
                            (box[0] + lw, box[1]),
                            box_color, cv2.FILLED,
                        )
                        cv2.putText(
                            annotated_frame, label,
                            (box[0], box[1] - baseline),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255) if (is_intruding or wl_match) else (0, 0, 0), 1,
                        )

                        # Watchlist match annotation
                        if wl_match:
                            wl_lines = [
                                f"WATCHLIST MATCH",
                                f"{wl_match['name']}",
                                f"{wl_match['status']}",
                            ]
                            y_offset = box[1] + lh + 5
                            for wl_line in wl_lines:
                                (wlw, wlh), wl_bl = cv2.getTextSize(
                                    wl_line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
                                )
                                cv2.rectangle(
                                    annotated_frame,
                                    (box[0], y_offset),
                                    (box[0] + wlw + 4, y_offset + wlh + wl_bl + 4),
                                    (0, 0, 200), cv2.FILLED,
                                )
                                cv2.putText(
                                    annotated_frame, wl_line,
                                    (box[0] + 2, y_offset + wlh + 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    (255, 255, 255), 2,
                                )
                                y_offset += wlh + wl_bl + 6

                        # Draw face bounding box if available from any face detection
                        fb = None
                        if track_id in self._latest_face_boxes:
                            fb = self._latest_face_boxes[track_id]
                        elif wl_match and wl_match.get("face_box"):
                            fb = wl_match.get("face_box")

                        if fb:
                            color = (0, 0, 255) if wl_match else (255, 0, 0)
                            label_fb = "MATCHED FACE" if wl_match else "FACE"
                            
                            cv2.rectangle(
                                annotated_frame,
                                (int(fb[0]), int(fb[1])),
                                (int(fb[2]), int(fb[3])),
                                color, 2,
                            )
                            # Add small label for face box
                            cv2.putText(
                                annotated_frame, label_fb,
                                (int(fb[0]), int(fb[1]) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                color, 1
                            )

                        # Draw trajectory
                        if modules["human_tracking"] and "trajectory" in det and len(det["trajectory"]) > 1:
                            traj = det["trajectory"]
                            for i in range(1, len(traj)):
                                cv2.line(annotated_frame, traj[i - 1], traj[i], (0, 255, 255), 2)

                # 5. Vehicle Detection
                if modules["vehicle_detection"]:
                    v_colors = {
                        "Car": (255, 128, 0),
                        "Motorcycle": (0, 128, 255),
                        "Bus": (255, 0, 255),
                        "Truck": (128, 255, 0),
                    }

                    for det in self._latest_vehicle_detections:
                        box = det["box"]
                        conf = det["conf"]
                        class_name = det["class_name"]
                        color = v_colors.get(class_name, (0, 255, 0))

                        cv2.rectangle(
                            annotated_frame,
                            (box[0], box[1]), (box[2], box[3]),
                            color, 2,
                        )

                        label = f"[{class_name}] {conf:.2f}"
                        (lw, lh), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                        )
                        cv2.rectangle(
                            annotated_frame,
                            (box[0], box[1] - lh - baseline),
                            (box[0] + lw, box[1]),
                            color, cv2.FILLED,
                        )
                        cv2.putText(
                            annotated_frame, label,
                            (box[0], box[1] - baseline),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                        )"""

content = re.sub(loop_pattern, new_loop_logic, content, flags=re.DOTALL)

# 4. Update the HUD and shared state blocks
# We also need to change `person_detections` to `self._latest_person_detections` in the bottom
# Replace `num_persons = len(person_detections)` with `num_persons = len(self._latest_person_detections)`
# Replace `num_vehicles = len(vehicle_detections)` with `num_vehicles = len(self._latest_vehicle_detections)`
content = content.replace("num_persons = len(person_detections)", "num_persons = len(self._latest_person_detections)")
content = content.replace("num_vehicles = len(vehicle_detections)", "num_vehicles = len(self._latest_vehicle_detections)")
content = content.replace("len(active_intrusions)", "len(self._latest_active_intrusions)")
content = content.replace("len(watchlist_matches)", "len(self._latest_watchlist_matches)")

# Also replace active_intrusions with self._latest_active_intrusions where used in HUD
content = content.replace("if active_intrusions and modules", "if self._latest_active_intrusions and modules")
content = content.replace("if watchlist_matches and modules", "if self._latest_watchlist_matches and modules")

# And in shared state:
content = content.replace("for det in vehicle_detections:", "for det in self._latest_vehicle_detections:")
content = content.replace("self._analytics[\"active_intrusions\"] = list(active_intrusions.keys())", "self._analytics[\"active_intrusions\"] = list(self._latest_active_intrusions.keys())")

# Update processing_fps to use self._ai_fps, and add stream_fps
fps_replace = """                    self._stream_fps = 1.0 / frame_time if frame_time > 0 else 0.0
                    self._analytics["processing_fps"] = round(self._ai_fps, 1)
                    self._analytics["stream_fps"] = round(self._stream_fps, 1)"""
content = re.sub(r'                    self._analytics\["processing_fps"\] = round\(inst_fps, 1\)', fps_replace, content)

# Change the FPS display in HUD to show AI and Stream FPS
hud_fps_replace = """                cv2.putText(annotated_frame, f"AI FPS: {self._ai_fps:.1f} | Stream FPS: {inst_fps:.1f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)"""
content = re.sub(r'                cv2\.putText\(annotated_frame, f"FPS: \{inst_fps:\.1f\}", \(20, 40\),\n.*?cv2\.FONT_HERSHEY_SIMPLEX, 0\.7, \(255, 255, 255\), 2\)', hud_fps_replace, content)

# Update active_tracks_info to use self._latest_active_tracks_info
content = content.replace("len(active_tracks_info)", "len(self._latest_active_tracks_info)")

# Update watchlist_matches.items() to self._latest_watchlist_matches.items()
content = content.replace("for tid, m in watchlist_matches.items()", "for tid, m in self._latest_watchlist_matches.items()")

with open("/Users/arshmeen/hackthon_mittul/sih26187-prototype/backend/app/services/frame_processor.py", "w") as f:
    f.write(content)
