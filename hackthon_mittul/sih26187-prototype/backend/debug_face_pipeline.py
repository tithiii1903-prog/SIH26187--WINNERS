import sys
import torch
import cv2
import numpy as np
from PIL import Image

from app.services.face_watchlist import FaceWatchlist
from app.services.detector import PersonDetector

print("Initializing FaceWatchlist...")
wl = FaceWatchlist()
wl._ensure_engine()

print("Initializing PersonDetector...")
# Note: PersonDetector might try to load yolov8n.pt
detector = PersonDetector(model_path="yolov8n.pt")

wl_id = list(wl._records.keys())[0] if wl._records else None
if not wl_id:
    print("No watchlist records found!")
    sys.exit(1)

img_path = f"config/watchlist_data/{wl_id}.jpg"
pt_path = f"config/watchlist_data/{wl_id}.pt"

stored_emb = torch.load(pt_path, map_location="cpu")
print(f"Stored embedding shape: {stored_emb.shape}, norm: {torch.norm(stored_emb).item()}")

# Load image via CV2
frame = cv2.imread(img_path)
h, w = frame.shape[:2]
print(f"Loaded image: {w}x{h}")

# 1. Detect person
print("Detecting person...")
detections = detector.detect(frame)
print(f"Detections: {len(detections)}")

# Fake a track ID so match_faces works
for i, d in enumerate(detections):
    d["id"] = i + 1

# 2. Run match_faces directly
print("Running match_faces...")
matches, events, face_boxes = wl.match_faces(frame, detections, frames_processed=0)
print("Matches:", matches)
print("Events:", events)
