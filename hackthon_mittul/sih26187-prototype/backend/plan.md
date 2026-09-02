# Implementation Plan

## Goal
Fix the spreading of Watchlist Matches (red boxes) to innocent people, and fix the lag in the CCTV MJPEG stream.

The root cause of the "red box leakage" is that the heavy AI processing (YOLO + MTCNN + ResNet) running on every frame causes the frame rate to drop so drastically that the ByteTrack tracker fails. When tracking fails due to large frame-to-frame displacement, ByteTrack incorrectly swaps Track IDs between people. An innocent person suddenly inherits the Track ID of a matched person, and thus gets painted red.

We will fix this by decoupling the AI processing cadence from the display cadence, and implementing independent per-track face processing schedules.

## Proposed Changes

### `frame_processor.py` (FrameProcessor)
1. **Decoupled Cadence**: Introduce `TARGET_AI_FPS` (e.g., 10 FPS). 
2. **Main Loop Mod**: The `_processing_loop` will read frames at `source_fps` (e.g., 30 FPS).
3. **AI Branch**: Only when `1.0 / TARGET_AI_FPS` seconds have passed, we will execute the AI pipeline (YOLO, ByteTrack, Face Watchlist, Vehicle Detection, Fence) on the current frame.
4. **Display Branch**: The current frame will be annotated using the *most recently computed* AI states (cached in variables like `latest_person_detections`, `latest_watchlist_matches`, etc.).
5. **Smoothing**: Between AI frames, bounding boxes will be drawn using the cached positions.
6. **Analytics**: We will track and report `processing_fps` (AI FPS) and `stream_fps`/`display_fps` (source loop FPS) separately in the HUD.

### `face_watchlist.py` (FaceWatchlist)
1. **Per-Track Scheduling**: Remove the global `FACE_PROCESS_INTERVAL` logic (`frames_processed % N`).
2. **Independent Processing**: For each active track, check if it's due for a face check using a per-track timer (e.g., check every 0.5 seconds if unmatched).
3. **Matched Track Optimization**: If a track is already matched, only run MTCNN detection to update the `face_box` location, and skip the ResNet embedding extraction. This guarantees that a matched track never blocks the face recognition schedule of newly appearing tracks.
4. **State Cleanup**: Ensure `_active_matches`, `_last_face_boxes`, and the new scheduling timers correctly evict disappeared tracks.

## Verification Plan
1. Start the backend and verify the MJPEG stream flows smoothly at the source FPS (~30 FPS), while AI updates at ~10 FPS.
2. Enroll a person. Have them and another non-enrolled person in the camera view.
3. Verify only the enrolled person gets a red box and "WATCHLIST MATCH" label.
4. Verify the non-enrolled person retains a green box.
5. Have the enrolled person leave the frame. Verify their match clears and does not transfer to the non-enrolled person.
6. Check that Virtual Fence and Vehicle Detection continue to function correctly at the new AI cadence.
