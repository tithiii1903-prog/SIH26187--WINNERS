import cv2
import numpy as np

img_full = cv2.imread("output/face_full.jpg")
img_crop = cv2.imread("output/face_crop.jpg")

diff = cv2.absdiff(img_full, img_crop)
print("Mean absolute difference (0-255):", np.mean(diff))

cv2.imwrite("output/face_diff.jpg", diff)
