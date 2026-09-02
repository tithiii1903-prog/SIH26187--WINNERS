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
detections = detector.detect(frame)
d = detections[0]
x1, y1, x2, y2 = d["box"]
h, w = frame.shape[:2]

for pad_factor in [0.0, 0.1, 0.5, 1.0, 2.0]:
    pad_x = int((x2 - x1) * pad_factor)
    pad_y = int((y2 - y1) * pad_factor)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    head_height = int((y2 - y1) * 0.5)
    cy2 = min(h, y1 + head_height + pad_y)
    
    crop = frame[cy1:cy2, cx1:cx2]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_crop = Image.fromarray(crop_rgb)
    
    face_tensor = wl._mtcnn(pil_crop)
    if face_tensor is not None:
        if face_tensor.dim() == 4:
            face_tensor = face_tensor[0]
        
        with torch.no_grad():
            face_input = face_tensor.unsqueeze(0).to(wl._device)
            new_emb = wl._resnet(face_input).cpu().squeeze()
            
        sim = torch.nn.functional.cosine_similarity(
            stored_emb.unsqueeze(0), new_emb.unsqueeze(0)
        ).item()
        print(f"Pad {pad_factor} -> Similarity: {sim}")
    else:
        print(f"Pad {pad_factor} -> Face not detected")
