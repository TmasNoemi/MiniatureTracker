import cv2
import numpy as np

# Setup Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def process_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:,:,2]
    frame = cv2.GaussianBlur(frame, (5, 5), 1)
    return frame

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ## The detector slides a small window across your image and at each position computes Haar features — these are just differences in brightness between rectangular regions
    ## feature value = (sum of pixels in white regions) - (sum of pixels in dark regions)
    #### a large positive value means a match -- there is a bright zone where the feature expects brightness
    ## A real face detector uses around 160,000 of these features. But rather than evaluating all 160,000 features on every window position, 
    ## the detector uses a cascade of stages — a sequence of increasingly strict filters

    # scaleFactor=1.1 — the detector slides a fixed-size window across the image, but faces can appear at any size. 
                        # So it actually runs multiple passes, each time making the detection window a bit larger. 
                        # scaleFactor is the growth multiplier between passes. With 1.1 you grow by 10% each pass: 10 passes to double in size. 
                        # With 1.3 you'd only need ~4 passes, but might miss faces between scale steps. Lower = more thorough but slower.
    # minNeighbors=5 — a face detection triggers many overlapping positive windows all around the same region. 
                        # This parameter says "I'll only report a detection if at least N neighboring windows also fired". 
                        # Faces generate dense clusters of hits; false positives (like your hand) generate sparse, isolated hits. 
                        # Raising this number is your main tool for filtering out hands.
    # minSize=(30, 30) — the smallest window (in pixels) that counts as a candidate face. 
                        # Any detection smaller than this is ignored outright. Useful for filtering distant false positives, 
                        # though not always helpful with hands since they're often large in frame.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=20, minSize=(30, 30))
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
        if len(eyes) >= 1:          # real faces almost always have detectable eyes
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, 'Face', (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return frame

def background_subtraction():
    cap = cv2.VideoCapture(0)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            preprocessed_frame = process_frame(frame)
            fgmask = fgbg.apply(frame)
            cv2.imshow('Background Subtraction', fgmask)
            if cv2.waitKey(1) == 27:
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


# The core idea: if you take two consecutive frames from a video, most pixels have moved slightly from one frame to the next. 
# Optical flow is the technique of estimating where each pixel (or point) went. 
# The result is a vector field — every tracked point gets an arrow showing its direction and speed of movement.
def optical_flow():
    # You don't track every single pixel — that would be slow and redundant. 
    # Instead you pick a set of good features to track, which in practice means corners. 
    # Corners are ideal because they have strong gradients in two directions, making them unambiguous to re-locate in the next frame
    feature_params = dict(maxCorners=200, qualityLevel=0.1, minDistance=7, blockSize=7)

    # Lucas-Kanade is the algorithm that actually computes where each point moved. 
    # The core assumption it makes is: the pixels in a small window around a point move the same way. 
    # So for each tracked point, it looks at a small patch and asks "where does this patch best fit in the next frame?"
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    cap = cv2.VideoCapture(0)

    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Impossibile leggere dalla webcam")

    ret, old_frame = cap.read()
    if not ret:
        raise RuntimeError("Errore lettura frame iniziale")

    h, w = old_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    rec = cv2.VideoWriter('output.avi', fourcc, 20.0, (w, h))

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)


        # calcOpticalFlowPyrLK takes the previous frame, the current frame, and the previous point positions p0, and returns:
        # p1 — the predicted new positions of every point
        # st — a status array: 1 if tracking succeeded for that point, 0 if it was lost (went offscreen, became occluded, etc.)
        if p0 is not None and len(p0) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            good_new = p1[st == 1] # (where they are now)
            good_old = p0[st == 1] # (where they were)
        else:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            good_new = np.array([])
            good_old = np.array([])

        old_gray = frame_gray.copy()

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = map(int, new.ravel())
            c, d = map(int, old.ravel())
            cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)
            cv2.line(frame, (a, b), (c, d), (0, 255, 0), 2)

        p0 = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else None

        # Rileva volti
        frame = detect_faces(frame)

        rec.write(frame)
        cv2.imshow("Tracked Motion", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    rec.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    optical_flow()