import sys
import torch
import cv2
from app.services.face_watchlist import FaceWatchlist
from app.services.detector import PersonDetector

print("Initializing FaceWatchlist...")
wl = FaceWatchlist()
wl._ensure_engine()
detector = PersonDetector(model_path="yolov8n.pt")

cap = cv2.VideoCapture("uploads/bg.mp4")
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    if frame_idx % 3 == 0:
        detections = detector.detect(frame)
        for i, d in enumerate(detections):
            d["id"] = i + 1
            
        matches, events = wl.match_faces(frame, detections, frames_processed=frame_idx)
        if matches or events:
            print(f"Frame {frame_idx}: Matches: {matches}")
            print(f"Frame {frame_idx}: Events: {events}")
    
    frame_idx += 1

cap.release()
print("Done processing dummy.mp4")
