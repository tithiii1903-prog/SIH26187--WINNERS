import sys
import torch
from PIL import Image
import cv2
import numpy as np
from app.services.face_watchlist import FaceWatchlist

# 1. Initialize
print("Initializing FaceWatchlist...")
wl = FaceWatchlist()
wl._ensure_engine()

# 2. Get the enrolled image path
wl_id = list(wl._records.keys())[0] if wl._records else None
if not wl_id:
    print("No watchlist records found!")
    sys.exit(1)

print(f"Using watchlist record: {wl_id}")
img_path = f"config/watchlist_data/{wl_id}.jpg"
pt_path = f"config/watchlist_data/{wl_id}.pt"

# 3. Load stored embedding
stored_emb = torch.load(pt_path, map_location="cpu")
print(f"Stored embedding shape: {stored_emb.shape}")
print(f"Stored embedding norm: {torch.norm(stored_emb).item()}")

# 4. Generate new embedding via Enrollment Path (RGB PIL)
pil_image = Image.open(img_path).convert("RGB")
# MTCNN
face_tensor_enroll = wl._mtcnn(pil_image)
if face_tensor_enroll.dim() == 4:
    face_tensor_enroll = face_tensor_enroll[0]

with torch.no_grad():
    face_input = face_tensor_enroll.unsqueeze(0).to(wl._device)
    new_emb_enroll = wl._resnet(face_input).cpu().squeeze()

print(f"New enrollment embedding shape: {new_emb_enroll.shape}")
print(f"New enrollment embedding norm: {torch.norm(new_emb_enroll).item()}")

# Compare
sim_enroll = torch.nn.functional.cosine_similarity(
    stored_emb.unsqueeze(0), new_emb_enroll.unsqueeze(0)
).item()
print(f"Similarity (Stored vs New Enrollment on exact same image): {sim_enroll}")


# 5. Generate new embedding via Match Path (BGR OpenCV)
cv_image = cv2.imread(img_path)
print(f"CV2 Image shape: {cv_image.shape}")

# Simulate match_faces logic
crop_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
pil_crop = Image.fromarray(crop_rgb)

face_tensor_match = wl._mtcnn(pil_crop)
if face_tensor_match.dim() == 4:
    face_tensor_match = face_tensor_match[0]

with torch.no_grad():
    face_input = face_tensor_match.unsqueeze(0).to(wl._device)
    new_emb_match = wl._resnet(face_input).cpu().squeeze()

print(f"New match embedding shape: {new_emb_match.shape}")
print(f"New match embedding norm: {torch.norm(new_emb_match).item()}")

sim_match = torch.nn.functional.cosine_similarity(
    stored_emb.unsqueeze(0), new_emb_match.unsqueeze(0)
).item()
print(f"Similarity (Stored vs New Match on exact same image): {sim_match}")

# Check difference between MTCNN outputs
print(f"MTCNN tensor difference (enroll vs match): {torch.abs(face_tensor_enroll - face_tensor_match).mean().item()}")
