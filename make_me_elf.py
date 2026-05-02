import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

# --- download the face landmarker model if not present ---
model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading face landmarker model...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        model_path
    )
    print("Done!")

# --- setup ---
RIGHT_EAR_PT  = 234
LEFT_EAR_PT = 454

ear_img = cv2.imread('elf_ear.png', cv2.IMREAD_UNCHANGED)

if ear_img is None:
    raise FileNotFoundError("elf_ear.png not found")
if ear_img.shape[2] != 4:
    raise ValueError("elf_ear.png must be RGBA (transparent background)")

def overlay_image(background, overlay, x, y):
    h, w = overlay.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, background.shape[1]), min(y + h, background.shape[0])
    if x1 >= x2 or y1 >= y2:
        return background
    ov = overlay[y1-y:y2-y, x1-x:x2-x]
    alpha = ov[:, :, 3:4] / 255.0
    bg_region = background[y1:y2, x1:x2]
    background[y1:y2, x1:x2] = (
        alpha * ov[:, :, :3] + (1 - alpha) * bg_region
    ).astype('uint8')
    return background

# --- mediapipe tasks setup ---
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO   # use VIDEO mode for webcam
)
landmarker = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

for _ in range(10):
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Impossibile leggere dalla webcam")  


show_ears = False
frame_idx = 0

print("Press E to toggle elf ears | Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_idx += 1

    # convert to mediapipe image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # detect landmarks
    results = landmarker.detect_for_video(mp_image, frame_idx)

    if show_ears and results.face_landmarks:
            lm = results.face_landmarks[0]
            h, w = frame.shape[:2]

            def pt(idx):
                return int(lm[idx].x * w), int(lm[idx].y * h)

            left_pt  = pt(LEFT_EAR_PT)
            right_pt = pt(RIGHT_EAR_PT)

            face_width = abs(right_pt[0] - left_pt[0])
            ear_size = int(face_width * 0.65)

            if ear_size > 0:
                ear_resized = cv2.resize(ear_img, (ear_size, ear_size))
                ear_flipped = cv2.flip(ear_resized, 1)

                overlap = ear_size // 4
                frame = overlay_image(frame, ear_resized,
                                    left_pt[0] - overlap,
                                    left_pt[1] - ear_size // 2)
                frame = overlay_image(frame, ear_flipped,
                                    right_pt[0] - ear_size + overlap,
                                    right_pt[1] - ear_size // 2)

            label = "Elf mode: ON" if show_ears else "Elf mode: OFF"
            color = (0, 255, 0) if show_ears else (0, 0, 255)
            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow('Elf Ears', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('e'):
        show_ears = not show_ears
    if key == ord('q'):
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()

        