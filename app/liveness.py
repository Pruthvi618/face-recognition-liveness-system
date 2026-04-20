import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:
    mp = None

# ---------- Mediapipe setup (if available) ----------
face_mesh = None
if mp is not None and hasattr(mp, "solutions"):
    face_mesh = mp.solutions.face_mesh.FaceMesh()

# OpenCV fallback for environments where mediapipe.solutions is unavailable
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
)
_eyes_open_frames = 0
_eyes_closed_frames = 0
_cooldown_frames = 0

# ---------- Blink detection ----------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye_indices):
    points = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_indices])

    vertical1 = np.linalg.norm(points[1] - points[5])
    vertical2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])

    if horizontal == 0:
        return 1.0

    return (vertical1 + vertical2) / (2.0 * horizontal)


def detect_blink(frame):
    if face_mesh is not None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return False

        landmarks = result.multi_face_landmarks[0].landmark

        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
        ear = (left_ear + right_ear) / 2.0
        return ear < 0.2

    global _eyes_open_frames, _eyes_closed_frames, _cooldown_frames
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        _eyes_open_frames = 0
        _eyes_closed_frames = 0
        _cooldown_frames = max(0, _cooldown_frames - 1)
        return False

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    roi_gray = gray[y : y + h, x : x + w]
    eyes = _eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=6)

    if _cooldown_frames > 0:
        _cooldown_frames -= 1
        return False

    if len(eyes) >= 1:
        _eyes_open_frames += 1
        _eyes_closed_frames = 0
        return False

    if _eyes_open_frames >= 2:
        _eyes_closed_frames += 1
        if 1 <= _eyes_closed_frames <= 4:
            _eyes_open_frames = 0
            _eyes_closed_frames = 0
            _cooldown_frames = 8
            return True

    return False


# ---------- Passive liveness ----------
def passive_liveness_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)

    blur_norm = min(blur_score / 100.0, 1.0)
    brightness_norm = 1.0 - abs(brightness - 127) / 127

    score = 0.6 * blur_norm + 0.4 * brightness_norm

    return score


# ---------- Hybrid liveness (THIS IS STEP 6.6) ----------
def check_liveness(frame):
    score = passive_liveness_score(frame)

    if score > 0.6:
        return True, "Passive passed"

    if score < 0.3:
        return False, "Passive failed"

    if detect_blink(frame):
        return True, "Active passed (blink)"

    return False, "Liveness failed"
