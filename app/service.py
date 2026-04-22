"""Application service layer for registration and verification flows.

Why this file exists:
- Coordinate model, database, similarity, and liveness modules.
- Provide clean high-level functions used by command-line scripts.
"""

from __future__ import annotations

import cv2
import time

from app.model import get_embedding
from app.database import ensure_schema, insert_user, get_all_users
from app.logger import get_logger
from app.utils import cosine_similarity
from app.liveness import (
    check_liveness,
    passive_liveness_score,
    anti_spoof_check,
    reset_liveness_state,
)
from config import settings

MIN_THRESHOLD = 0.65
MAX_THRESHOLD = 0.82
STD_MULTIPLIER = 1.5
MIN_MARGIN = 0.04
LIVENESS_TIMEOUT_SECONDS = settings.liveness_timeout_seconds
MIN_PASSIVE_FOR_MATCH = settings.min_passive_for_match

LOGGER = get_logger(__name__)
_schema_ready = False


def _ensure_schema_ready() -> None:
    """Ensure DB schema exists before any read/write operation."""
    global _schema_ready
    if not _schema_ready:
        ensure_schema()
        _schema_ready = True


def register_user(name: str, image_path: str) -> str:
    """Register one user from an image path.

    Why we need this:
    - Converts the image into an embedding and stores it for future matching.
    """
    _ensure_schema_ready()
    embedding = get_embedding(image_path)
    if embedding is None:
        return "No face detected"

    insert_user(name, embedding)
    return "User registered successfully"


def _adaptive_threshold(scores: list[float]) -> float:
    """Compute a bounded threshold from the current similarity distribution."""
    if not scores:
        return MIN_THRESHOLD

    mean_score = sum(scores) / len(scores)
    variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
    std_score = variance ** 0.5
    threshold = mean_score + (STD_MULTIPLIER * std_score)
    return max(MIN_THRESHOLD, min(threshold, MAX_THRESHOLD))


def _match_embedding(embedding: list[float]) -> str:
    """Match a probe embedding against enrolled users.

    Why we need this:
    - Centralize ranking, thresholding, and ambiguity handling in one function.
    """
    _ensure_schema_ready()
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


def verify_user(image_path: str) -> str:
    """Verify identity from a still image path.

    Why we need this:
    - Basic verification flow for scripts and batch-style checks.
    """
    embedding = get_embedding(image_path)
    if embedding is None:
        return "No face detected"

    return _match_embedding(embedding)


def verify_with_liveness(image_path: str) -> str:
    """Verify identity from an image only if liveness passes first.

    Why we need this:
    - Prevent simple spoof attempts from being treated as valid face matches.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        return "Invalid image"

    anti_spoof_ok, _ = anti_spoof_check(frame)
    if (not anti_spoof_ok) or passive_liveness_score(frame) < MIN_PASSIVE_FOR_MATCH:
        return "Liveness failed (use realtime blink + head movement for strong check)"

    return verify_user(image_path)


def verify_realtime(camera_index: int = 0) -> str:
    """Run webcam liveness and then perform real-time identity matching.

    Why we need this:
    - This is the most practical flow for login-like scenarios.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return "Camera unavailable"

    reset_liveness_state()
    print("Checking liveness... Do blink and slight head movement")
    last_msg = None
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            cv2.destroyAllWindows()
            return "Camera read failed"

        is_live, msg = check_liveness(frame)
        LOGGER.debug("Liveness status: %s", msg)

        if msg != last_msg:
            print(msg)
            last_msg = msg

        if is_live:
            break

        if time.time() - start_time > LIVENESS_TIMEOUT_SECONDS:
            cap.release()
            cv2.destroyAllWindows()
            return "Liveness timeout, please try again"

        cv2.imshow("Liveness Check", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return "Cancelled"

    print("Liveness passed. Hold still and face the camera...")

    # After challenge steps (blink/turn), collect a short burst of frames and
    # try matching each one. This avoids matching on a side-pose frame.
    matched_result = None
    sampled = 0
    max_samples = 24

    while sampled < max_samples:
        ret, confirm_frame = cap.read()
        if not ret:
            break
        sampled += 1

        if sampled % 2 != 0:
            continue

        embedding = get_embedding(confirm_frame, enforce_detection=False)
        if embedding is None:
            continue

        result = _match_embedding(embedding)
        if result.startswith("Matched:"):
            matched_result = result
            break

    cap.release()
    cv2.destroyAllWindows()

    if matched_result is not None:
        LOGGER.info("Realtime verification success: %s", matched_result)
        return matched_result

    LOGGER.info("Realtime verification result: Unknown person")
    return "Unknown person"
