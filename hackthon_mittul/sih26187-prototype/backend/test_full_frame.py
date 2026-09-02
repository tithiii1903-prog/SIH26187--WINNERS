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
pt_path = f"config/watchlist_data/{wl_id}.pt"
stored_emb = torch.load(pt_path, map_location="cpu")

frame = cv2.imread(img_path)
h, w = frame.shape[:2]

# Detect person
detections = detector.detect(frame)
person_box = detections[0]["box"] # Assuming 1st detection is our person

# Convert FULL frame to RGB PIL
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
pil_frame = Image.fromarray(frame_rgb)

# Detect faces in FULL frame
boxes, probs = wl._mtcnn.detect(pil_frame)
face_tensor = wl._mtcnn(pil_frame)

if face_tensor is not None and boxes is not None:
    for i, box in enumerate(boxes):
        # Calculate intersection with person_box
        x1 = max(box[0], person_box[0])
        y1 = max(box[1], person_box[1])
        x2 = min(box[2], person_box[2])
        y2 = min(box[3], person_box[3])
        if x2 > x1 and y2 > y1:
            print(f"Face {i} intersects with person box!")
            
            # Use this face tensor
            ft = face_tensor[i]
            
            with torch.no_grad():
                face_input = ft.unsqueeze(0).to(wl._device)
                new_emb = wl._resnet(face_input).cpu().squeeze()
            
            sim = torch.nn.functional.cosine_similarity(
                stored_emb.unsqueeze(0), new_emb.unsqueeze(0)
            ).item()
            print(f"Similarity using FULL FRAME MTCNN: {sim}")
