import re

with open("/Users/arshmeen/hackthon_mittul/sih26187-prototype/backend/app/services/frame_processor.py", "r") as f:
    content = f.read()

prefix = content[:content.find("    def _processing_loop(self):")]

new_method = """    def _processing_loop(self):
        \"\"\"Main processing loop — runs in background thread.\"\"\"
        try:
            if not self.video_source.open():
                with self._lock:
                    self._status = "ERROR"
                    self._error_message = "Failed to open video source"
                return

            source_fps = self.video_source.get_fps()
            frame_interval = 1.0 / source_fps if source_fps > 0 else 1.0 / 30.0

            with self._lock:
                self._status = "LIVE"
                self._analytics["source_fps"] = source_fps

            frames_processed = 0
            source_read_failed = False
            last_ai_time = 0.0
            ai_interval = 1.0 / self.TARGET_AI_FPS

            while not self._stop_event.is_set():
                frame_start = time.time()

                with self._lock:
                    if self._fence_reload_requested:
                        self._fence_reload_requested = False
                        self.virtual_fence.reload_zones(self.fence_config_path)

                success, frame = self.video_source.read_frame()
                if not success:
                    source_read_failed = True
                    break

                timestamp = frames_processed / source_fps if source_fps > 0 else 0.0

                with self._lock:
                    modules = self._modules_enabled.copy()

                annotated_frame = frame.copy()
                new_events = []
                current_time = time.time()

                # --- AI Processing Cadence ---
                if current_time - last_ai_time >= ai_interval:
                    last_ai_time = current_time
                    ai_start = time.time()

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

                    current_ids = set()
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

                    if modules["virtual_fence"] and modules["human_detection"]:
                        fence_events = self.virtual_fence.process_frame(
                            self._latest_person_detections, timestamp
                        )
                        new_events.extend(fence_events)
                        self._latest_active_intrusions = self.virtual_fence.get_active_intrusions()
                    else:
                        self._latest_active_intrusions = {}

                    if modules["vehicle_detection"]:
                        self._latest_vehicle_detections = self.vehicle_detector.detect(frame)
                    else:
                        self._latest_vehicle_detections = []

                    ai_time = time.time() - ai_start
                    self._ai_fps = 1.0 / ai_time if ai_time > 0 else 0.0

                # --- Draw Latest States on Current Frame ---
                if modules["virtual_fence"] and modules["human_detection"]:
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

                if modules["human_detection"]:
                    for det in self._latest_person_detections:
                        box = det["box"]
                        conf = det["conf"]
                        track_id = det.get("id")

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
                            cv2.putText(
                                annotated_frame, label_fb,
                                (int(fb[0]), int(fb[1]) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                color, 1
                            )

                        if modules["human_tracking"] and "trajectory" in det and len(det["trajectory"]) > 1:
                            traj = det["trajectory"]
                            for i in range(1, len(traj)):
                                cv2.line(annotated_frame, traj[i - 1], traj[i], (0, 255, 255), 2)

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
                        )

                frame_time = time.time() - frame_start
                self._stream_fps = 1.0 / frame_time if frame_time > 0 else 0.0

                num_persons = len(self._latest_person_detections)
                num_vehicles = len(self._latest_vehicle_detections)

                cv2.putText(annotated_frame, f"AI FPS: {self._ai_fps:.1f} | Stream FPS: {self._stream_fps:.1f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Persons: {num_persons}", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(annotated_frame, f"Vehicles: {num_vehicles}", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)

                hud_y = 130
                if self._latest_active_intrusions and modules["virtual_fence"]:
                    cv2.putText(
                        annotated_frame,
                        f"ACTIVE INTRUSIONS: {len(self._latest_active_intrusions)}",
                        (20, hud_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                    )
                    hud_y += 30

                if self._latest_watchlist_matches and modules["face_watchlist"]:
                    cv2.putText(
                        annotated_frame,
                        f"WATCHLIST MATCHES: {len(self._latest_watchlist_matches)}",
                        (20, hud_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                    )

                v_counts = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0}
                for det in self._latest_vehicle_detections:
                    cn = det["class_name"]
                    if cn in v_counts:
                        v_counts[cn] += 1

                intrusion_entries = sum(1 for e in new_events if e["type"] == "INTRUSION_ENTER")
                intrusion_exits = sum(1 for e in new_events if e["type"] == "INTRUSION_EXIT")

                with self._lock:
                    self._latest_frame = annotated_frame

                    self._analytics["current_persons"] = num_persons
                    self._analytics["active_tracks"] = len(self._latest_active_tracks_info)
                    if num_persons > self._analytics["peak_persons"]:
                        self._analytics["peak_persons"] = num_persons
                    self._analytics["current_vehicles"] = num_vehicles
                    if num_vehicles > self._analytics["peak_vehicles"]:
                        self._analytics["peak_vehicles"] = num_vehicles
                    self._analytics["cars"] = v_counts["Car"]
                    self._analytics["motorcycles"] = v_counts["Motorcycle"]
                    self._analytics["buses"] = v_counts["Bus"]
                    self._analytics["trucks"] = v_counts["Truck"]
                    self._analytics["active_intrusions"] = list(self._latest_active_intrusions.keys())
                    self._analytics["total_intrusion_entries"] += intrusion_entries
                    self._analytics["total_intrusion_exits"] += intrusion_exits
                    self._analytics["processing_fps"] = round(self._ai_fps, 1)
                    self._analytics["stream_fps"] = round(self._stream_fps, 1)
                    self._analytics["frames_processed"] = frames_processed + 1
                    self._analytics["timestamp"] = round(timestamp, 2)
                    self._analytics["active_watchlist_matches"] = [
                        {
                            "track_id": tid,
                            "name": m["name"],
                            "status": m["status"],
                            "similarity": m.get("similarity", 0),
                            "wl_id": m.get("wl_id"),
                        }
                        for tid, m in self._latest_watchlist_matches.items()
                    ]
                    if len(self._latest_active_tracks_info) > self._analytics["max_active_tracks"]:
                        self._analytics["max_active_tracks"] = len(self._latest_active_tracks_info)

                    for ev in new_events:
                        self._events.append(ev)

                self._frame_ready.set()
                self._frame_ready.clear()

                frames_processed += 1

                elapsed = time.time() - frame_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    self._stop_event.wait(timeout=sleep_time)

            with self._lock:
                if self._status == "LIVE":
                    if source_read_failed and self.video_source.is_live():
                        self._status = "ERROR"
                        self._error_message = "Camera disconnected or unavailable"
                    else:
                        self._status = "STOPPED"

        except Exception as e:
            with self._lock:
                self._status = "ERROR"
                self._error_message = str(e)
            import traceback
            traceback.print_exc()

        finally:
            self.video_source.release()
"""

with open("/Users/arshmeen/hackthon_mittul/sih26187-prototype/backend/app/services/frame_processor.py", "w") as f:
    f.write(prefix + new_method)
