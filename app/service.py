import cv2

from app.model import get_embedding
from app.database import insert_user, get_all_users
from app.utils import cosine_similarity
from app.liveness import check_liveness

MIN_THRESHOLD = 0.65
MAX_THRESHOLD = 0.82
STD_MULTIPLIER = 1.5
MIN_MARGIN = 0.04


def register_user(name, image_path):
    embedding = get_embedding(image_path)
    if embedding is None:
        return "No face detected"

    insert_user(name, embedding)
    return "User registered successfully"


def _adaptive_threshold(scores):
    """Compute a bounded threshold from the current similarity distribution."""
    if not scores:
        return MIN_THRESHOLD

    mean_score = sum(scores) / len(scores)
    variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
    std_score = variance ** 0.5
    threshold = mean_score + (STD_MULTIPLIER * std_score)
    return max(MIN_THRESHOLD, min(threshold, MAX_THRESHOLD))


def _match_embedding(embedding):
    users = get_all_users()
    if not users:
        return "No registered users found"

    scored_by_name = {}
    for name, db_embedding in users:
        score = cosine_similarity(embedding, db_embedding)
        previous_best = scored_by_name.get(name, -1)
        if score > previous_best:
            scored_by_name[name] = score

    scored_users = sorted(scored_by_name.items(), key=lambda item: item[1], reverse=True)
    best_match, best_score = scored_users[0]
    second_best_score = scored_users[1][1] if len(scored_users) > 1 else -1

    threshold = _adaptive_threshold([score for _, score in scored_users])
    margin = best_score - second_best_score if second_best_score >= 0 else best_score

    if best_score >= threshold and margin >= MIN_MARGIN:
        return f"Matched: {best_match} (score={best_score:.2f})"
    return "Unknown person"


def verify_user(image_path):
    embedding = get_embedding(image_path)
    if embedding is None:
        return "No face detected"

    return _match_embedding(embedding)


def verify_with_liveness(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        return "Invalid image"

    is_live, _ = check_liveness(frame)
    if not is_live:
        return "Liveness failed"

    return verify_user(image_path)


def verify_realtime():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Camera unavailable"

    print("Checking liveness...")
    last_msg = None

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            cv2.destroyAllWindows()
            return "Camera read failed"

        is_live, msg = check_liveness(frame)
        if msg != last_msg:
            print(msg)
            last_msg = msg

        if is_live:
            break

        cv2.imshow("Liveness Check", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return "Cancelled"

    cap.release()
    cv2.destroyAllWindows()

    # Use the captured frame directly to avoid disk I/O.
    embedding = get_embedding(frame, enforce_detection=False)
    if embedding is None:
        return "No face detected"

    return _match_embedding(embedding)
