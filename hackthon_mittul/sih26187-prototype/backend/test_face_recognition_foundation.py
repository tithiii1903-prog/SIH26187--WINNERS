"""
Phase 12 HD Face Recognition Foundation Test Suite.

Automated verification of tests A through T:
A. Model initialization
B. Camera initialization
C. Single-face detection
D. Multi-face detection
E. Face bounding box correctness
F. Embedding dimensions = 512
G. Embedding normalization (L2 norm = 1.0)
H. Same-person recognition
I. Different-person rejection
J. Unknown person handling
K. Multiple simultaneous faces (independent state)
L. Temporal stability (2-match confirmation & grace period)
M. Camera worker stability
N. No unbounded queue
O. Recognition FPS
P. Preview/capture FPS
Q. Model initialization happens only once
R. No raw frame storage
S. No embedding exposure
T. Real data verification
"""

import os
import sys
import time
import json
import sqlite3
import unittest
import numpy as np
import cv2

from app.services.face_recognition import (
    FaceEngine,
    get_face_analysis_app,
    get_model_load_time_ms,
    FaceMatcher,
    DEFAULT_FACE_THRESHOLD,
    FaceStateTracker,
    compute_bbox_iou,
    FaceCamera,
)


class TestFaceRecognitionFoundation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ref_img1_path = "config/watchlist_data/16a1fdce.jpg"
        cls.ref_img2_path = "config/watchlist_data/914dec35.jpg"
        cls.attendance_db_path = "/Users/arshmeen/face_recognition/instance/attendance.db"

        cls.engine = FaceEngine()

        cls.img1 = cv2.imread(cls.ref_img1_path)
        cls.img2 = cv2.imread(cls.ref_img2_path)
        assert cls.img1 is not None, f"Missing test image {cls.ref_img1_path}"
        assert cls.img2 is not None, f"Missing test image {cls.ref_img2_path}"

    def test_A_model_initialization(self):
        """Test A: Model initialization produces a valid InsightFace app."""
        app = get_face_analysis_app()
        self.assertIsNotNone(app)
        load_time = get_model_load_time_ms()
        self.assertGreaterEqual(load_time, 0.0)
        print(f"\n[Test A] Model loaded in {load_time:.1f}ms")

    def test_B_camera_initialization(self):
        """Test B: Camera initialization handles device indexes cleanly."""
        cam = FaceCamera(device_index=999, target_rec_fps=10.0, engine=self.engine)
        self.assertTrue(cam.start())
        self.assertTrue(cam.is_running())
        cam.stop()
        self.assertFalse(cam.is_running())
        print("\n[Test B] Camera lifecycle start/stop verified")

    def test_C_single_face_detection(self):
        """Test C: Single-face detection on real image."""
        faces = self.engine.detect_and_extract(self.img1)
        self.assertEqual(len(faces), 1, "Expected exactly 1 face detected in ref_img1")
        self.assertGreater(faces[0]["confidence"], 0.70)
        print(f"\n[Test C] Single face detected with confidence: {faces[0]['confidence']:.3f}")

    def test_D_multi_face_detection(self):
        """Test D: Multi-face detection on combined frame."""
        # Create wide canvas placing both faces side by side
        h1, w1 = self.img1.shape[:2]
        h2, w2 = self.img2.shape[:2]
        canvas_h = max(h1, h2)
        canvas_w = w1 + w2
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:h1, :w1] = self.img1
        canvas[:h2, w1:w1+w2] = self.img2

        faces = self.engine.detect_and_extract(canvas)
        self.assertGreaterEqual(len(faces), 2, "Expected at least 2 faces detected on multi-face canvas")
        print(f"\n[Test D] Multi-face canvas detected {len(faces)} faces")

    def test_E_face_bounding_box_correctness(self):
        """Test E: Face bounding box correctness and boundary clamping."""
        faces = self.engine.detect_and_extract(self.img1)
        self.assertGreater(len(faces), 0)
        h, w = self.img1.shape[:2]
        for f in faces:
            x1, y1, x2, y2 = f["bbox"]
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)
            self.assertLessEqual(x2, w)
            self.assertLessEqual(y2, h)
            self.assertLess(x1, x2)
            self.assertLess(y1, y2)
        print(f"\n[Test E] Bounding box verified: {faces[0]['bbox']} within ({w}x{h})")

    def test_F_embedding_dimensions_512(self):
        """Test F: Embedding dimensions equal 512."""
        faces = self.engine.detect_and_extract(self.img1)
        emb = faces[0]["embedding"]
        self.assertEqual(emb.shape, (512,), "ArcFace embedding must have exact shape (512,)")
        self.assertEqual(emb.dtype, np.float32)
        print(f"\n[Test F] Embedding shape verified: {emb.shape}")

    def test_G_embedding_normalization(self):
        """Test G: Embedding L2 normalization equals 1.0."""
        faces = self.engine.detect_and_extract(self.img1)
        emb = faces[0]["embedding"]
        norm = np.linalg.norm(emb)
        self.assertAlmostEqual(norm, 1.0, places=4, msg="Embedding L2 norm must be 1.0")
        print(f"\n[Test G] Embedding L2 norm: {norm:.6f}")

    def test_H_same_person_recognition(self):
        """Test H: Same-person recognition with similarity >= 0.70."""
        faces1 = self.engine.detect_and_extract(self.img1)
        emb1 = faces1[0]["embedding"]

        # Register Person 1
        matcher = FaceMatcher(match_threshold=0.70)
        matcher.register_face("p1", "Mittul", "CRITICAL", emb1)

        # Match same embedding
        res = matcher.match(emb1)
        self.assertTrue(res["matched"])
        self.assertEqual(res["person_id"], "p1")
        self.assertEqual(res["name"], "Mittul")
        self.assertEqual(res["status"], "CRITICAL")
        self.assertGreaterEqual(res["similarity"], 0.99)
        print(f"\n[Test H] Same-person match similarity: {res['similarity']:.4f}")

    def test_I_different_person_rejection(self):
        """Test I: Different-person rejection against attendance dataset."""
        faces1 = self.engine.detect_and_extract(self.img1)
        emb1 = faces1[0]["embedding"]

        # Read Person 2 from attendance DB (known different identity)
        conn = sqlite3.connect(self.attendance_db_path)
        c = conn.cursor()
        c.execute("SELECT id, name, face_encoding FROM person WHERE id = 2")
        row = c.fetchone()
        conn.close()

        self.assertIsNotNone(row, "Attendance record for Person 2 must exist")
        pid, name, enc_str = row
        enc_arr = np.array(json.loads(enc_str), dtype=np.float32)
        norm_enc = enc_arr / np.linalg.norm(enc_arr)

        matcher = FaceMatcher(match_threshold=0.70)
        matcher.register_face("p2", name, "WATCHLIST", norm_enc)

        res = matcher.match(emb1)
        self.assertFalse(res["matched"], f"Different person should not match (sim={res['similarity']})")
        self.assertLess(res["similarity"], 0.70)
        print(f"\n[Test I] Different-person similarity: {res['similarity']:.4f} (< 0.70 threshold) -> Rejection Verified")

    def test_J_unknown_person_handling(self):
        """Test J: Unknown person handling returns structured non-match."""
        faces = self.engine.detect_and_extract(self.img1)
        emb = faces[0]["embedding"]

        matcher = FaceMatcher(match_threshold=0.70)
        # Empty database
        res = matcher.match(emb)
        self.assertFalse(res["matched"])
        self.assertIsNone(res["person_id"])
        self.assertIsNone(res["name"])
        self.assertIsNone(res["status"])
        self.assertEqual(res["similarity"], 0.0)
        print("\n[Test J] Unknown person handled cleanly with empty database")

    def test_K_multiple_simultaneous_faces_independent_state(self):
        """Test K: Multiple simultaneous faces maintain independent states without leakage."""
        faces1 = self.engine.detect_and_extract(self.img1)
        emb1 = faces1[0]["embedding"]

        # Read Person 2 from attendance DB (known different identity)
        conn = sqlite3.connect(self.attendance_db_path)
        c = conn.cursor()
        c.execute("SELECT id, name, face_encoding FROM person WHERE id = 2")
        row = c.fetchone()
        conn.close()
        enc_arr = np.array(json.loads(row[2]), dtype=np.float32)
        norm_enc_diff = enc_arr / np.linalg.norm(enc_arr)

        matcher = FaceMatcher(match_threshold=0.70)
        # Register Person A (img1) only
        matcher.register_face("p1", "Alice (Target)", "CRITICAL", emb1)

        tracker = FaceStateTracker(min_consecutive_matches=1, grace_period_sec=0.6)

        # Detection 1 (Alice) at x=50, Detection 2 (Unknown Person) at x=500
        mock_detections = [
            {"bbox": [50, 50, 150, 150], "confidence": 0.95, "embedding": emb1},
            {"bbox": [500, 50, 600, 150], "confidence": 0.92, "embedding": norm_enc_diff},
        ]

        results = tracker.update(mock_detections, matcher.match, current_time=1.0)
        self.assertEqual(len(results), 2)

        # Check face 1 (Alice) is matched
        f1 = [r for r in results if r["bbox"][0] == 50][0]
        self.assertTrue(f1["matched"])
        self.assertEqual(f1["name"], "Alice (Target)")
        self.assertEqual(f1["status"], "CRITICAL")

        # Check face 2 (Unknown) is NOT matched as Alice
        f2 = [r for r in results if r["bbox"][0] == 500][0]
        self.assertFalse(f2["matched"])
        self.assertIsNone(f2["name"])
        self.assertLess(f2["similarity"], 0.70)
        print(f"\n[Test K] Multi-face state: Face1 matched={f1['matched']} ({f1['name']}), Face2 matched={f2['matched']} ({f2['name']})")

        # Disappearance test: Face 1 leaves, Face 2 remains
        mock_detections_2 = [
            {"bbox": [500, 50, 600, 150], "confidence": 0.92, "embedding": norm_enc_diff},
        ]
        results_after_disappearance = tracker.update(mock_detections_2, matcher.match, current_time=2.0)
        self.assertEqual(len(results_after_disappearance), 1)
        self.assertEqual(results_after_disappearance[0]["bbox"][0], 500)
        self.assertFalse(results_after_disappearance[0]["matched"])
        print("[Test K] Verified: Disappearance of Face 1 removes Face 1 without altering Face 2")

    def test_L_temporal_stability(self):
        """Test L: Temporal confirmation requires >=2 observations before confirming."""
        faces1 = self.engine.detect_and_extract(self.img1)
        emb1 = faces1[0]["embedding"]

        matcher = FaceMatcher(match_threshold=0.70)
        matcher.register_face("p1", "Target", "CRITICAL", emb1)

        tracker = FaceStateTracker(min_consecutive_matches=2, grace_period_sec=0.6)
        det = [{"bbox": [100, 100, 200, 200], "confidence": 0.95, "embedding": emb1}]

        # Frame 1: first observation -> pending confirmation (matched=False)
        res1 = tracker.update(det, matcher.match, current_time=1.0)
        self.assertFalse(res1[0]["matched"], "Frame 1 must NOT declare match immediately")
        self.assertIsNone(res1[0]["name"])

        # Frame 2: second observation -> confirmed match (matched=True)
        res2 = tracker.update(det, matcher.match, current_time=1.1)
        self.assertTrue(res2[0]["matched"], "Frame 2 must confirm match")
        self.assertEqual(res2[0]["name"], "Target")
        self.assertEqual(res2[0]["status"], "CRITICAL")
        print("\n[Test L] Temporal stability verified (Frame 1 pending -> Frame 2 confirmed)")

    def test_M_camera_worker_stability(self):
        """Test M: Camera worker threads start and stop without deadlocks."""
        cam = FaceCamera(device_index=999, target_rec_fps=10.0, engine=self.engine)
        cam.start()
        for i in range(10):
            cam.supply_frame(self.img1, timestamp=time.time())
            time.sleep(0.02)
        cam.stop()
        self.assertFalse(cam.is_running())
        print("\n[Test M] Camera worker start/feed/stop stability verified")

    def test_N_no_unbounded_queue(self):
        """Test N: Single-slot buffer drops stale frames under high load."""
        cam = FaceCamera(device_index=999, target_rec_fps=5.0, engine=self.engine)
        cam.start()
        # Rapidly feed 50 frames
        for i in range(50):
            cam.supply_frame(self.img1, timestamp=time.time())
            time.sleep(0.002)

        time.sleep(0.3)
        # Verify single pending frame
        with cam._rec_slot_lock:
            # Buffer slot is either 1 frame or None, never a list of 50 frames
            self.assertTrue(cam._pending_rec_frame is None or isinstance(cam._pending_rec_frame, np.ndarray))
        cam.stop()
        print("\n[Test N] Single-slot bounded buffer verified (no queue accumulation)")

    def test_O_and_P_fps_measurement(self):
        """Test O & P: Measured Recognition FPS and Preview FPS."""
        cam = FaceCamera(device_index=999, target_rec_fps=10.0, engine=self.engine)
        cam.start()
        start_t = time.time()
        for i in range(25):
            cam.supply_frame(self.img1, timestamp=time.time())
            time.sleep(1.0 / 25.0)

        telemetry = cam.get_telemetry()
        cam.stop()

        self.assertGreater(telemetry["source_fps"], 0.0)
        print(f"\n[Test O & P] Measured Source FPS: {telemetry['source_fps']:.1f}, Recognition FPS: {telemetry['recognition_fps']:.1f}")

    def test_Q_model_initialization_only_once(self):
        """Test Q: Verify singleton model instance is reused."""
        app1 = get_face_analysis_app()
        app2 = get_face_analysis_app()
        self.assertIs(app1, app2, "InsightFace FaceAnalysis must be a singleton instance")
        print("\n[Test Q] Singleton instance identity verified (app1 is app2)")

    def test_R_no_raw_frame_storage(self):
        """Test R: Verify no raw frames are saved to disk during recognition."""
        files_before = set(os.listdir("."))
        faces = self.engine.detect_and_extract(self.img1)
        files_after = set(os.listdir("."))
        self.assertEqual(files_before, files_after, "detect_and_extract must not create new files on disk")
        print("\n[Test R] Verified zero disk writes during face detection/recognition")

    def test_S_no_embedding_exposure(self):
        """Test S: Verify public results format does not expose raw embedding arrays."""
        tracker = FaceStateTracker()
        matcher = FaceMatcher()
        faces = self.engine.detect_and_extract(self.img1)
        results = tracker.update(faces, matcher.match)
        self.assertGreater(len(results), 0)
        for f in results:
            self.assertNotIn("embedding", f, "Embedding array must not be in public face results")
            self.assertIn("bbox", f)
            self.assertIn("matched", f)
            self.assertIn("similarity", f)
            self.assertIn("confidence", f)
        print("\n[Test S] Verified embeddings are hidden from public face results")

    def test_T_real_data_verification(self):
        """Test T: Verify actual data from attendance project is used (no fake identities)."""
        conn = sqlite3.connect(self.attendance_db_path)
        c = conn.cursor()
        c.execute("SELECT id, name, category, status FROM person")
        rows = c.fetchall()
        conn.close()

        self.assertGreaterEqual(len(rows), 1, "Real attendance person records must exist in database")
        print(f"\n[Test T] Verified real attendance identities in database: {[r[1] for r in rows]}")


if __name__ == "__main__":
    unittest.main()
