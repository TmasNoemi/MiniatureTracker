"""
Visualizzatore di landmark: mostra tutti i punti con il loro indice.
Usa i tasti per filtrare la zona di interesse:
  A/D  - sposta il filtro X (colonna)
  W/S  - sposta il filtro Y (riga)
  +/-  - allarga/restringe la finestra di visualizzazione
  0    - mostra tutti i landmark (nessun filtro)
  Q    - esci
"""
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os, urllib.request

model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        model_path
    )

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO,
)
landmarker = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
for _ in range(5):
    cap.read()

frame_idx = 0
# Zona di interesse: mostra solo i landmark con x in [cx-r, cx+r] e y in [cy-r, cy+r]
# Valori normalizzati [0,1]. Parte su "tutto lo schermo".
cx, cy, r = 0.5, 0.5, 0.5  # filtro iniziale: tutto visibile

print("A/D = muovi filtro X | W/S = muovi filtro Y | +/- = zoom | 0 = tutto | Q = esci")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_idx += 1
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = landmarker.detect_for_video(mp_image, frame_idx)

    if results.face_landmarks:
        lm = results.face_landmarks[0]
        for i, p in enumerate(lm):
            if abs(p.x - cx) <= r and abs(p.y - cy) <= r:
                px = int(p.x * w)
                py = int(p.y * h)
                cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)
                cv2.putText(frame, str(i), (px + 2, py - 2),
                            cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 255, 0), 1)

    # Mostra il rettangolo del filtro attuale
    x1 = int((cx - r) * w)
    y1 = int((cy - r) * h)
    x2 = int((cx + r) * w)
    y2 = int((cy + r) * h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 120, 255), 1)
    cv2.putText(frame, f"cx={cx:.2f} cy={cy:.2f} r={r:.2f}",
                (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 120, 255), 1)

    cv2.imshow('Ear Landmark Finder', frame)

    key = cv2.waitKey(1) & 0xFF
    step = 0.03
    if key == ord('a'):
        cx = max(0, cx - step)
    elif key == ord('d'):
        cx = min(1, cx + step)
    elif key == ord('w'):
        cy = max(0, cy - step)
    elif key == ord('s'):
        cy = min(1, cy + step)
    elif key == ord('+') or key == ord('='):
        r = max(0.05, r - step)
    elif key == ord('-'):
        r = min(0.5, r + step)
    elif key == ord('0'):
        cx, cy, r = 0.5, 0.5, 0.5
    elif key == ord('q'):
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()
