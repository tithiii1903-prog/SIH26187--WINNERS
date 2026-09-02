import torch
import torchvision
import ultralytics
from ultralytics import YOLO
import cv2
import numpy as np

print("--- 1. Python Package Verification ---")
print(f"PyTorch: {torch.__version__}")
print(f"Torchvision: {torchvision.__version__}")
print(f"Ultralytics: {ultralytics.__version__}")
print(f"OpenCV: {cv2.__version__}")

print("\n--- 2. PyTorch MPS Verification ---")
mps_built = torch.backends.mps.is_built()
mps_avail = torch.backends.mps.is_available()
print(f"MPS Built: {mps_built} | MPS Available: {mps_avail}")

print("\n--- 3. YOLO Model Load Test ---")
model = YOLO('yolov8n.pt')
print("Model loaded successfully.")
print(f"Device: {next(model.model.parameters()).device}")

print("\n--- 4. OpenCV Video Read Test ---")
out = cv2.VideoWriter('dummy.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (640, 480))
for _ in range(5):
    out.write(np.zeros((480, 640, 3), dtype=np.uint8))
out.release()
cap = cv2.VideoCapture('dummy.mp4')
ret, frame = cap.read()
if ret and frame is not None:
    print("OpenCV video read passed.")
else:
    print("OpenCV video read failed.")
cap.release()
