import sys
import torch
import cv2
import numpy as np
from PIL import Image
from app.services.face_watchlist import FaceWatchlist
from app.services.detector import PersonDetector

wl = FaceWatchlist()
wl._ensure_engine()
detector = PersonDetector(model_path="yolov8n.pt")

wl_id = list(wl._records.keys())[0]
img_path = f"config/watchlist_data/{wl_id}.jpg"

frame = cv2.imread(img_path)
detections = detector.detect(frame)
# Let's simulate a tracking sequence
for i, det in enumerate(detections):
    det["id"] = i + 1 # assign different IDs
    det["trajectory"] = []
    
# Duplicate the first detection to make track 2 have a face
detections[1] = dict(detections[0])
detections[1]["id"] = 2
# Offset it a bit so they aren't completely identical
detections[1]["box"] = [b + 50 for b in detections[0]["box"]]

print("Frame 1")
matches, events, face_boxes = wl.match_faces(frame, detections, frames_processed=0)
print("Matches 1:", matches)
print("Face Boxes 1:", face_boxes)

print("\nFrame 4 (should use debounce)")
matches, events, face_boxes = wl.match_faces(frame, detections, frames_processed=3)
print("Matches 4:", matches)
print("Face Boxes 4:", face_boxes)

print("\nFrame 7 (track left)")
matches, events, face_boxes = wl.match_faces(frame, [], frames_processed=6)
print("Matches 7:", matches)
print("Face Boxes 7:", face_boxes)
