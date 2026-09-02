"""
HD Face Recognition Test Runner & Demonstration Harness.

Runs the isolated HD Face Recognition Foundation pipeline and reports real measured metrics:
- Model load time
- Resolution & Capture FPS
- Recognition FPS & Latency
- Single and multi-face detections
- Live cosine matching against registered faces
"""

import sys
import time
import os
import cv2
import numpy as np

from app.services.face_recognition import (
    FaceEngine,
    FaceMatcher,
    FaceStateTracker,
    FaceCamera,
    get_model_load_time_ms
)


def run_benchmark(duration_sec: float = 3.0, device_index: int = 0):
    print("=" * 60)
    print("PHASE 12 HD FACE RECOGNITION FOUNDATION BENCHMARK")
    print("=" * 60)

    # 1. Initialize Engine
    t0 = time.time()
    engine = FaceEngine()
    load_time_ms = get_model_load_time_ms()
    print(f"[1] FaceEngine Loaded: {load_time_ms:.1f}ms")

    # 2. Initialize Matcher & Register Reference Faces
    matcher = FaceMatcher(match_threshold=0.70)
    ref_path_1 = "config/watchlist_data/16a1fdce.jpg"
    if os.path.exists(ref_path_1):
        img_ref = cv2.imread(ref_path_1)
        if img_ref is not None:
            faces = engine.detect_and_extract(img_ref)
            if faces:
                matcher.register_face(
                    person_id="target_001",
                    name="Mittul (Registered)",
                    status="CRITICAL",
                    embedding=faces[0]["embedding"]
                )
                print(f"[2] Registered reference face from {ref_path_1} (512D ArcFace)")

    # 3. Initialize State Tracker
    state_tracker = FaceStateTracker(min_consecutive_matches=2, grace_period_sec=0.6)
    print("[3] FaceStateTracker initialized (2-frame temporal confirmation, 0.6s grace)")

    # 4. Initialize Camera Service
    camera = FaceCamera(
        device_index=device_index,
        target_rec_fps=10.0,
        match_threshold=0.70,
        engine=engine,
        matcher=matcher,
        state_tracker=state_tracker
    )

    print(f"[4] Starting FaceCamera service (device={device_index})...")
    camera.start()

    # If camera hardware not available, feed sample test frames
    if camera._cap is None or not camera._cap.isOpened():
        print("    [Notice] Hardware camera not open. Supplying sample video frames to test harness.")
        sample_img = cv2.imread(ref_path_1) if os.path.exists(ref_path_1) else np.zeros((720, 1280, 3), dtype=np.uint8)
        start_t = time.time()
        frame_idx = 0
        while time.time() - start_t < duration_sec:
            camera.supply_frame(sample_img, timestamp=time.time())
            time.sleep(1.0 / 30.0)
            frame_idx += 1
    else:
        print(f"    Running camera recognition for {duration_sec}s...")
        time.sleep(duration_sec)

    telemetry = camera.get_telemetry()
    results = camera.get_latest_results()

    camera.stop()

    print("-" * 60)
    print("MEASURED PERFORMANCE REPORT:")
    print(f"  Source Capture FPS:            {telemetry['source_fps']}")
    print(f"  Recognition Cadence FPS:       {telemetry['recognition_fps']}")
    print(f"  Avg Recognition Latency:       {telemetry['average_recognition_latency_ms']} ms")
    print(f"  Detected Faces in Last Frame:  {telemetry['number_of_faces']}")
    print(f"  Model Load Time:               {telemetry['model_load_time_ms']} ms")
    print(f"  Match Threshold:               {telemetry['match_threshold']}")
    print(f"  Registered Faces:              {telemetry['registered_faces_count']}")
    if results.get("faces"):
        for i, f in enumerate(results["faces"]):
            print(f"    Face {i+1}: bbox={f['bbox']}, matched={f['matched']}, name={f['name']}, sim={f['similarity']:.3f}, conf={f['confidence']:.3f}")
    print("=" * 60)
    return telemetry, results


if __name__ == "__main__":
    dev_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_benchmark(duration_sec=2.0, device_index=dev_idx)
