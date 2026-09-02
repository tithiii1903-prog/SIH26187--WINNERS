import cv2
import numpy as np
import torch
from PIL import Image
from app.services.face_watchlist import FaceWatchlist
from app.services.detector import PersonDetector

wl = FaceWatchlist()
wl._ensure_engine()
detector = PersonDetector(model_path="yolov8n.pt")

wl_id = list(wl._records.keys())[0]
img_path = f"config/watchlist_data/{wl_id}.jpg"
frame = cv2.imread(img_path)
pil_full = Image.open(img_path).convert("RGB")

# 1. Full image face tensor
face_tensor_full = wl._mtcnn(pil_full)
if face_tensor_full.dim() == 4:
    face_tensor_full = face_tensor_full[0]

# Convert tensor to image for saving
img_full = face_tensor_full.permute(1, 2, 0).numpy()
img_full = ((img_full + 1) * 127.5).astype(np.uint8)
cv2.imwrite("output/face_full.jpg", cv2.cvtColor(img_full, cv2.COLOR_RGB2BGR))

# 2. Person crop face tensor
detections = detector.detect(frame)
d = detections[0]
x1, y1, x2, y2 = d["box"]
h, w = frame.shape[:2]
pad_x = int((x2 - x1) * 0.1)
pad_y = int((y2 - y1) * 0.05)
cx1 = max(0, x1 - pad_x)
cy1 = max(0, y1 - pad_y)
cx2 = min(w, x2 + pad_x)
head_height = int((y2 - y1) * 0.5)
cy2_head = min(h, y1 + head_height + pad_y)

crop = frame[cy1:cy2_head, cx1:cx2]
cv2.imwrite("output/person_crop.jpg", crop)

crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
pil_crop = Image.fromarray(crop_rgb)

face_tensor_crop = wl._mtcnn(pil_crop)
if face_tensor_crop.dim() == 4:
    face_tensor_crop = face_tensor_crop[0]

img_crop = face_tensor_crop.permute(1, 2, 0).numpy()
img_crop = ((img_crop + 1) * 127.5).astype(np.uint8)
cv2.imwrite("output/face_crop.jpg", cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR))

diff = torch.abs(face_tensor_full - face_tensor_crop).mean().item()
print(f"Tensor difference: {diff}")
