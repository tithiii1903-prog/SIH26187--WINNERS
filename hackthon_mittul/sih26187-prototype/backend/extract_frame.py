import cv2
import os

os.makedirs("output", exist_ok=True)
cap = cv2.VideoCapture("dummy.mp4")

# Read a frame around 2 seconds in
cap.set(cv2.CAP_PROP_POS_MSEC, 2000)
ret, frame = cap.read()
if ret:
    cv2.imwrite("output/sample_frame.jpg", frame)
    print("Saved output/sample_frame.jpg")
else:
    print("Could not read frame")

cap.release()
