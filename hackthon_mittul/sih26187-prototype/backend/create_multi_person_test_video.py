import cv2
import numpy as np
import os
from PIL import Image

def generate_multi_person_video(output_path="multi_person_test.mp4", duration_sec=6, fps=30):
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Load enrolled face image
    ref_face_path = "config/watchlist_data/16a1fdce.jpg"
    if os.path.exists(ref_face_path):
        ref_face_img = cv2.imread(ref_face_path)
    else:
        ref_face_img = np.full((120, 100, 3), (180, 180, 220), dtype=np.uint8)

    # Let's create a realistic silhouette / person template for Person A and Person B
    # Person A template: height 360, width 140
    # Person B template: height 360, width 140
    # Person C template: height 360, width 140

    def draw_person(canvas, x_center, y_bottom, is_target=False, label="A"):
        # Person height ~360, width ~130
        h = 360
        w = 130
        x1 = int(x_center - w // 2)
        y1 = int(y_bottom - h)
        x2 = int(x_center + w // 2)
        y2 = int(y_bottom)

        # Body / Clothes
        torso_color = (80, 120, 180) if is_target else (150, 100, 80)
        pants_color = (40, 40, 40)
        skin_color = (160, 190, 220)

        # Draw legs / pants
        cv2.rectangle(canvas, (x1 + 15, y1 + 180), (x1 + 55, y2), pants_color, cv2.FILLED)
        cv2.rectangle(canvas, (x2 - 55, y1 + 180), (x2 - 15, y2), pants_color, cv2.FILLED)

        # Draw torso / jacket
        cv2.rectangle(canvas, (x1 + 10, y1 + 90), (x2 - 10, y1 + 190), torso_color, cv2.FILLED)
        # Arms
        cv2.rectangle(canvas, (x1, y1 + 95), (x1 + 12, y1 + 180), torso_color, cv2.FILLED)
        cv2.rectangle(canvas, (x2 - 12, y1 + 95), (x2, y1 + 180), torso_color, cv2.FILLED)

        # Head / Face
        head_w = 70
        head_h = 80
        hx1 = int(x_center - head_w // 2)
        hy1 = int(y1 + 10)
        hx2 = int(x_center + head_w // 2)
        hy2 = int(hy1 + head_h)

        if is_target and ref_face_img is not None:
            # Resize ref face to fit head area
            resized_face = cv2.resize(ref_face_img, (head_w, head_h))
            canvas[hy1:hy2, hx1:hx2] = resized_face
        else:
            # Unknown person face (simple oval + features)
            cv2.ellipse(canvas, (x_center, hy1 + head_h // 2), (head_w // 2, head_h // 2), 0, 0, 360, skin_color, cv2.FILLED)
            cv2.circle(canvas, (x_center - 12, hy1 + 35), 4, (30, 30, 30), cv2.FILLED)
            cv2.circle(canvas, (x_center + 12, hy1 + 35), 4, (30, 30, 30), cv2.FILLED)
            cv2.ellipse(canvas, (x_center, hy1 + 55), (12, 6), 0, 0, 180, (50, 50, 120), 2)
            # Hair
            cv2.ellipse(canvas, (x_center, hy1 + 15), (head_w // 2, 20), 0, 180, 360, (20, 20, 20), cv2.FILLED)

    total_frames = duration_sec * fps
    bg_color = (210, 215, 220)

    for f in range(total_frames):
        frame = np.full((height, width, 3), bg_color, dtype=np.uint8)
        # Add floor line
        cv2.line(frame, (0, 600), (width, 600), (160, 160, 160), 3)

        # Phase 1: Frames 0 to 60 (Person A at x=250, Person B at x=800)
        # Phase 2: Frames 60 to 120 (Person A moves from 250 -> 600, Person B moves from 800 -> 950)
        # Phase 3: Frames 120 to 180 (Person A disappears/exits frame, Person B at x=900, Person C enters at x=200)
        if f < 60:
            draw_person(frame, x_center=250, y_bottom=600, is_target=True, label="A")
            draw_person(frame, x_center=800, y_bottom=600, is_target=False, label="B")
        elif f < 120:
            progress = (f - 60) / 60.0
            pos_a = int(250 + (600 - 250) * progress)
            pos_b = int(800 + (950 - 800) * progress)
            draw_person(frame, x_center=pos_a, y_bottom=600, is_target=True, label="A")
            draw_person(frame, x_center=pos_b, y_bottom=600, is_target=False, label="B")
        else:
            # Person A has left the scene
            progress = (f - 120) / 60.0
            pos_c = int(100 + 200 * progress)
            draw_person(frame, x_center=900, y_bottom=600, is_target=False, label="B")
            draw_person(frame, x_center=pos_c, y_bottom=600, is_target=False, label="C")

        out.write(frame)

    out.release()
    print(f"Generated test video: {output_path} ({total_frames} frames, {width}x{height} @ {fps}fps)")

if __name__ == "__main__":
    generate_multi_person_video()
