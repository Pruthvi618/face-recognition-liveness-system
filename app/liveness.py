"""Liveness checks used before identity verification.

Why this file exists:
- Reduce spoofing risk by checking for signs of a live person.
- Combine passive signals (image quality) with active blink detection.
"""

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

# OpenCV fallback for environments where mediapipe is unavailable.
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
)
_eyes_open_frames = 0
_eyes_closed_frames = 0
_cooldown_frames = 0
_mp_eyes_open_frames = 0
_mp_eyes_closed_frames = 0
_mp_cooldown_frames = 0
_mp_ear_baseline = None
_mp_prev_nose_x = None
_mp_nose_x_history = []
_mp_head_baseline_x = None
_haar_prev_face_center_x = None
_haar_head_baseline_x = None
_anti_prev_gray = None
_anti_suspicious_streak = 0
_blink_seen = False
_movement_seen = False
PASSIVE_HARD_FAIL_THRESHOLD = 0.15
PASSIVE_SOFT_PASS_THRESHOLD = 0.6
MP_MIN_OPEN_EAR = 0.22
MP_BASELINE_ALPHA = 0.08
MP_BLINK_DROP_RATIO = 0.84
MP_MOVEMENT_WINDOW = 12
MP_MOVEMENT_RANGE_THRESHOLD = 0.01
HAAR_MOVEMENT_PX_THRESHOLD = 8
MP_TURN_DELTA_X = 0.035
HAAR_TURN_DELTA_PX = 18
ANTI_SPOOF_STREAK_THRESHOLD = 10
ANTI_BANDING_THRESHOLD = 0.55
ANTI_FLICKER_THRESHOLD = 0.10

# ---------- Blink detection ----------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye_indices):
    """Compute eye openness ratio from selected eye landmark points.

    Why we need this:
    - EAR drops when eyelids close, so it is useful for blink detection.
    """
    points = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_indices])

    vertical1 = np.linalg.norm(points[1] - points[5])
    vertical2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])

    if horizontal == 0:
        return 1.0

    return (vertical1 + vertical2) / (2.0 * horizontal)


def detect_blink(frame):
    """Return True when the current frame indicates a blink.

    Why we need this:
    - A blink is a simple active liveness cue that static photos cannot provide.
    """
    if face_mesh is not None:
        global _mp_eyes_open_frames, _mp_eyes_closed_frames, _mp_cooldown_frames, _mp_ear_baseline
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            _mp_eyes_open_frames = 0
            _mp_eyes_closed_frames = 0
            _mp_cooldown_frames = max(0, _mp_cooldown_frames - 1)
            _mp_ear_baseline = None
            return False

        landmarks = result.multi_face_landmarks[0].landmark

        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
        ear = (left_ear + right_ear) / 2.0
        if _mp_ear_baseline is None:
            _mp_ear_baseline = ear

        # Update baseline mostly when eyes appear open to avoid learning blink values.
        if ear >= max(MP_MIN_OPEN_EAR, _mp_ear_baseline * 0.9):
            _mp_ear_baseline = (1.0 - MP_BASELINE_ALPHA) * _mp_ear_baseline + MP_BASELINE_ALPHA * ear

        if _mp_cooldown_frames > 0:
            _mp_cooldown_frames -= 1
            return False

        open_threshold = max(MP_MIN_OPEN_EAR, _mp_ear_baseline * 0.95)
        closed_threshold = min(0.23, _mp_ear_baseline * MP_BLINK_DROP_RATIO)

        if ear >= open_threshold:
            _mp_eyes_open_frames += 1
            _mp_eyes_closed_frames = 0
            return False

        if _mp_eyes_open_frames >= 1 and ear <= closed_threshold:
            _mp_eyes_closed_frames += 1
            if 1 <= _mp_eyes_closed_frames <= 4:
                _mp_eyes_open_frames = 0
                _mp_eyes_closed_frames = 0
                _mp_cooldown_frames = 6
                return True

        # If Mediapipe does not detect a blink, try OpenCV fallback too.
        return _detect_blink_haar(frame)

    return _detect_blink_haar(frame)


def detect_head_movement(frame):
    """Detect simple active liveness via small horizontal head movement.

    Why we need this:
    - Blink can fail on some cameras/angles.
    - Slight head turn is an easy fallback challenge for real users.
    """
    global _mp_prev_nose_x, _mp_nose_x_history, _haar_prev_face_center_x

    if face_mesh is not None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            _mp_prev_nose_x = None
            _mp_nose_x_history = []
        else:
            landmarks = result.multi_face_landmarks[0].landmark
            nose_x = landmarks[1].x

            if _mp_prev_nose_x is None:
                _mp_prev_nose_x = nose_x
                _mp_nose_x_history = [nose_x]
            else:
                _mp_prev_nose_x = nose_x
                _mp_nose_x_history.append(nose_x)
                if len(_mp_nose_x_history) > MP_MOVEMENT_WINDOW:
                    _mp_nose_x_history = _mp_nose_x_history[-MP_MOVEMENT_WINDOW:]

            if len(_mp_nose_x_history) >= 4:
                movement_range = max(_mp_nose_x_history) - min(_mp_nose_x_history)
                if movement_range >= MP_MOVEMENT_RANGE_THRESHOLD:
                    _mp_nose_x_history = []
                    return True

    # Fallback movement detector from face box center shift.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        _haar_prev_face_center_x = None
        return False

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    center_x = x + (w // 2)
    if _haar_prev_face_center_x is None:
        _haar_prev_face_center_x = center_x
        return False

    delta_px = abs(center_x - _haar_prev_face_center_x)
    _haar_prev_face_center_x = center_x
    return delta_px >= HAAR_MOVEMENT_PX_THRESHOLD


def detect_head_turn(frame):
    """Detect left/right head turn direction for challenge-response liveness.

    Returns:
    - "left" or "right" when direction is confidently detected.
    - None when no reliable turn is detected.
    """
    global _mp_head_baseline_x, _haar_head_baseline_x

    if face_mesh is not None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        if result.multi_face_landmarks:
            nose_x = result.multi_face_landmarks[0].landmark[1].x
            if _mp_head_baseline_x is None:
                _mp_head_baseline_x = nose_x
                return None

            # Keep a stable neutral baseline to reduce drift.
            _mp_head_baseline_x = (0.98 * _mp_head_baseline_x) + (0.02 * nose_x)
            delta = nose_x - _mp_head_baseline_x
            if delta >= MP_TURN_DELTA_X:
                return "right"
            if delta <= -MP_TURN_DELTA_X:
                return "left"
        else:
            _mp_head_baseline_x = None

    # Haar fallback using face-center displacement.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        _haar_head_baseline_x = None
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    center_x = x + (w // 2)
    if _haar_head_baseline_x is None:
        _haar_head_baseline_x = center_x
        return None

    _haar_head_baseline_x = int((0.98 * _haar_head_baseline_x) + (0.02 * center_x))
    delta_px = center_x - _haar_head_baseline_x
    if delta_px >= HAAR_TURN_DELTA_PX:
        return "right"
    if delta_px <= -HAAR_TURN_DELTA_PX:
        return "left"

    return None


def reset_liveness_state():
    """Reset detector state between sessions.

    Why we need this:
    - Stateful counters/baselines should not leak between verification attempts.
    """
    global _eyes_open_frames, _eyes_closed_frames, _cooldown_frames
    global _mp_eyes_open_frames, _mp_eyes_closed_frames, _mp_cooldown_frames
    global _mp_ear_baseline, _mp_prev_nose_x, _mp_nose_x_history
    global _mp_head_baseline_x, _haar_prev_face_center_x, _haar_head_baseline_x
    global _anti_prev_gray, _anti_suspicious_streak, _blink_seen, _movement_seen

    _eyes_open_frames = 0
    _eyes_closed_frames = 0
    _cooldown_frames = 0
    _mp_eyes_open_frames = 0
    _mp_eyes_closed_frames = 0
    _mp_cooldown_frames = 0
    _mp_ear_baseline = None
    _mp_prev_nose_x = None
    _mp_nose_x_history = []
    _mp_head_baseline_x = None
    _haar_prev_face_center_x = None
    _haar_head_baseline_x = None
    _anti_prev_gray = None
    _anti_suspicious_streak = 0
    _blink_seen = False
    _movement_seen = False


def anti_spoof_check(frame):
    """Heuristic replay/screen artifact check.

    Returns:
    - (True, "Anti-spoof passed") for normal-looking camera frames.
    - (False, "...") when repeated screen/replay artifacts are detected.

    Why we need this:
    - Challenge-response alone can still be replayed with pre-recorded videos.
    - This hook blocks repeated display-like artifact patterns.
    """
    global _anti_prev_gray, _anti_suspicious_streak

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    row_signal = np.mean(gray, axis=1)
    col_signal = np.mean(gray, axis=0)

    row_fft = np.abs(np.fft.rfft(row_signal - np.mean(row_signal)))
    col_fft = np.abs(np.fft.rfft(col_signal - np.mean(col_signal)))

    def _highfreq_ratio(spectrum):
        if spectrum.size < 8:
            return 0.0
        split = max(1, int(0.65 * spectrum.size))
        total = np.sum(spectrum) + 1e-6
        high = np.sum(spectrum[split:])
        return float(high / total)

    banding_score = max(_highfreq_ratio(row_fft), _highfreq_ratio(col_fft))

    flicker_score = 0.0
    if _anti_prev_gray is not None:
        delta = np.abs(gray - _anti_prev_gray)
        flicker_score = float(np.mean(delta) / 255.0)
    _anti_prev_gray = gray

    suspicious_now = banding_score >= ANTI_BANDING_THRESHOLD and flicker_score <= ANTI_FLICKER_THRESHOLD
    if suspicious_now:
        _anti_suspicious_streak += 1
    else:
        _anti_suspicious_streak = max(0, _anti_suspicious_streak - 1)

    if _anti_suspicious_streak >= ANTI_SPOOF_STREAK_THRESHOLD:
        return False, "Replay/screen artifacts detected"

    return True, "Anti-spoof passed"


def _detect_blink_haar(frame):
    """Fallback blink detector using Haar cascades.

    Why we need this:
    - Some cameras/angles work poorly with one detector.
    - Running a second detector improves pass reliability for real users.
    """
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
    """Score image quality heuristics that correlate with real camera capture.

    Why we need this:
    - Blur and extreme brightness are common in spoofed/replayed inputs.
    - A fast passive score filters easy failures before active checks.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)

    blur_norm = min(blur_score / 100.0, 1.0)
    brightness_norm = 1.0 - abs(brightness - 127) / 127

    score = 0.6 * blur_norm + 0.4 * brightness_norm

    return score


# ---------- Hybrid liveness  ----------
def check_liveness(frame):
    """Strict realtime liveness: anti-spoof + blink + head movement.

    Why we need this:
    - Replayed mobile videos can contain blink and movement-like patterns.
    - Combining anti-spoof artifact checks with two active cues is stronger.
    """
    global _blink_seen, _movement_seen

    anti_spoof_ok, anti_spoof_msg = anti_spoof_check(frame)
    if not anti_spoof_ok:
        return False, anti_spoof_msg

    score = passive_liveness_score(frame)
    if score < PASSIVE_HARD_FAIL_THRESHOLD:
        return False, "Face quality too low, improve lighting"

    if detect_blink(frame):
        _blink_seen = True

    if detect_head_movement(frame):
        _movement_seen = True

    if _blink_seen and _movement_seen:
        return True, "Liveness passed (blink + head movement)"

    if _blink_seen and not _movement_seen:
        return False, "Blink detected. Move head slightly left/right"

    if _movement_seen and not _blink_seen:
        return False, "Head movement detected. Blink required"

    return False, "Do blink and slight head movement"
