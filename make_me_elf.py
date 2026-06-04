import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading face landmarker model...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        model_path
    )

# MediaPipe non traccia il padiglione auricolare — usiamo i punti del face oval
# più vicini all'orecchio, confermati visivamente con find_ear_landmarks.py:
#   schermo sinistro: cima orecchio ≈ 127, mid orecchio ≈ 234
#   schermo destro:   cima orecchio ≈ 356, mid orecchio ≈ 454
# (nota: in MediaPipe la numerazione segue la prospettiva della telecamera,
#  quindi 127/234 sono i landmark "destra" del modello ma appaiono a sinistra
#  nell'immagine specchio)
L_EAR_TOP = 234   # cima orecchio sinistro (schermo)
L_EAR_MID = 93   # mid orecchio sinistro  (schermo)
R_EAR_TOP = 454   # cima orecchio destro   (schermo)
R_EAR_MID = 323   # mid orecchio destro    (schermo)

ELF_ELONGATION    = 2.0   # altezza della punta rispetto all'altezza orecchio
ELF_OUTWARD_SHIFT = 0.28  # spostamento del centro verso l'esterno (frazione semi-larghezza viso)
# sigma alla base della punta (largo = colpisce qpiù area dell'orecchio)
ELF_SIGMA_BASE    = 0.65
# sigma alla punta (stretto = crea la forma appuntita senza tirare verso il viso)
ELF_SIGMA_TIP     = 0.08


def build_elf_warp_maps(h, w, ear_cx, ear_cy, ear_top_y, out_dir):
    """
    Costruisce le mappe inverse (map_x, map_y) per cv2.remap.

    out_dir: -1 per orecchio sinistro (esterno = sinistra),
             +1 per orecchio destro  (esterno = destra).

    La forma a punta viene creata restringendo il sigma orizzontale
    man mano che si sale (largo alla base, stretto alla punta).
    Non c'è spostamento orizzontale (disp_x=0): questo elimina l'effetto
    di "tirare" il contenuto verso il viso.
    """
    ear_h     = max(ear_cy - ear_top_y, 1)
    extension = ear_h * ELF_ELONGATION

    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    map_x, map_y = np.meshgrid(xs, ys)

    dx = map_x - ear_cx

    # t = 0 a ear_top_y (base), t = 1 alla punta dell'elfo, t < 0 sotto
    t        = (ear_top_y - map_y) / extension
    t_active = np.maximum(t, 0.0)
    taper    = np.where(t > 1.0, np.exp(-((t - 1.0) * 4.0) ** 2), 1.0)

    # sigma si restringe al crescere di t → la punta è naturalmente appuntita
    sigma_out = ear_h * (ELF_SIGMA_BASE * (1.0 - t_active) + ELF_SIGMA_TIP * t_active)
    sigma_in  = ear_h * 0.08  # strettissimo verso il viso per non distorcerlo

    is_outward = (dx * out_dir) >= 0
    sigma_x    = np.where(is_outward, sigma_out, sigma_in)
    h_falloff  = np.exp(-dx**2 / (2.0 * sigma_x**2))

    influence = t_active * taper * h_falloff
    disp_y    = extension * influence  # solo spostamento verticale, niente disp_x

    return map_x.astype(np.float32), (map_y + disp_y).astype(np.float32)


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
for _ in range(10):
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read from webcam")

show_ears = False
debug     = False   # premi D per vedere i punti di controllo
frame_idx = 0
print("E = toggle elf ears | D = toggle debug | Q = esci")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_idx += 1

    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results  = landmarker.detect_for_video(mp_image, frame_idx)

    if results.face_landmarks:
        lm   = results.face_landmarks[0]
        h, w = frame.shape[:2]

        def pt(idx):
            return (int(lm[idx].x * w), int(lm[idx].y * h))

        l_top = pt(L_EAR_TOP)
        l_mid = pt(L_EAR_MID)
        r_top = pt(R_EAR_TOP)
        r_mid = pt(R_EAR_MID)

        if show_ears:
            # sposta il centro del warp leggermente verso l'esterno rispetto al
            # face oval, così colpisce l'orecchio e non il viso
            face_half_w = max(abs(r_top[0] - l_top[0]) // 2, 1)
            offset = int(face_half_w * ELF_OUTWARD_SHIFT)

            # orecchio sinistro (schermo): esterno = sinistra → out_dir = -1
            mx, my = build_elf_warp_maps(
                h, w, l_top[0] - offset, l_mid[1], l_top[1], out_dir=-1)
            frame = cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

            # orecchio destro (schermo): esterno = destra → out_dir = +1
            mx, my = build_elf_warp_maps(
                h, w, r_top[0] + offset, r_mid[1], r_top[1], out_dir=+1)
            frame = cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        if debug:
            for p, label in [(l_top, "L_TOP 127"), (l_mid, "L_MID 234"),
                             (r_top, "R_TOP 356"), (r_mid, "R_MID 454")]:
                cv2.circle(frame, p, 4, (0, 255, 255), -1)
                cv2.putText(frame, label, (p[0] + 5, p[1] - 5),
                            cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 255), 1)

    label = "Elf: ON" if show_ears else "Elf: OFF"
    color = (0, 255, 0) if show_ears else (0, 0, 255)
    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('Elf Ears', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('e'):
        show_ears = not show_ears
    elif key == ord('d'):
        debug = not debug
    elif key == ord('q'):
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()
