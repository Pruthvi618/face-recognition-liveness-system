"""Application service layer for registration and verification flows.

Why this file exists:
- Coordinate model, database, similarity, and liveness modules.
- Provide clean high-level functions used by command-line scripts.
"""

from __future__ import annotations

import cv2
import time
from typing import Any

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


def register_user_from_frame(name: str, frame: Any) -> dict[str, Any]:
    """Register one user from an already-decoded image frame."""
    _ensure_schema_ready()
    embedding = get_embedding(frame)
    if embedding is None:
        return {
            "success": False,
            "status": "no_face_detected",
            "message": "No face detected",
            "name": name,
        }

    insert_user(name, embedding)
    return {
        "success": True,
        "status": "registered",
        "message": "User registered successfully",
        "name": name,
    }


def _adaptive_threshold(scores: list[float]) -> float:
    """Compute a bounded threshold from the current similarity distribution."""
    if not scores:
        return MIN_THRESHOLD

    mean_score = sum(scores) / len(scores)
    variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
    std_score = variance ** 0.5
    threshold = mean_score + (STD_MULTIPLIER * std_score)
    return max(MIN_THRESHOLD, min(threshold, MAX_THRESHOLD))


def _match_embedding_result(embedding: list[float]) -> dict[str, Any]:
    """Match a probe embedding against enrolled users.

    Why we need this:
    - Centralize ranking, thresholding, and ambiguity handling in one function.
    """
    _ensure_schema_ready()
    users = get_all_users()
    if not users:
        return {
            "success": False,
            "status": "no_registered_users",
            "message": "No registered users found",
            "matched": False,
            "name": None,
            "score": None,
            "threshold": None,
            "margin": None,
        }

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
        return {
            "success": True,
            "status": "matched",
            "message": f"Matched: {best_match} (score={best_score:.2f})",
            "matched": True,
            "name": best_match,
            "score": round(best_score, 4),
            "threshold": round(threshold, 4),
            "margin": round(margin, 4),
        }

    return {
        "success": True,
        "status": "unknown",
        "message": "Unknown person",
        "matched": False,
        "name": None,
        "score": round(best_score, 4),
        "threshold": round(threshold, 4),
        "margin": round(margin, 4),
    }


def _match_embedding(embedding: list[float]) -> str:
    """Return the legacy CLI message for an embedding match."""
    return _match_embedding_result(embedding)["message"]


def verify_user(image_path: str) -> str:
    """Verify identity from a still image path.

    Why we need this:
    - Basic verification flow for scripts and batch-style checks.
    """
    embedding = get_embedding(image_path)
    if embedding is None:
        return "No face detected"

    return _match_embedding(embedding)


def verify_user_from_frame(frame: Any) -> dict[str, Any]:
    """Verify identity from an already-decoded image frame."""
    embedding = get_embedding(frame)
    if embedding is None:
        return {
            "success": False,
            "status": "no_face_detected",
            "message": "No face detected",
            "matched": False,
            "name": None,
            "score": None,
            "threshold": None,
            "margin": None,
        }

    return _match_embedding_result(embedding)


def verify_frame_with_liveness(frame: Any) -> dict[str, Any]:
    """Verify identity from a decoded frame only if image liveness passes."""
    anti_spoof_ok, anti_spoof_msg = anti_spoof_check(frame)
    passive_score = passive_liveness_score(frame)
    if (not anti_spoof_ok) or passive_score < MIN_PASSIVE_FOR_MATCH:
        return {
            "success": False,
            "status": "liveness_failed",
            "message": "Liveness failed (use realtime blink + head movement for strong check)",
            "liveness": {
                "passed": False,
                "anti_spoof_message": anti_spoof_msg,
                "passive_score": round(passive_score, 4),
                "minimum_passive_score": MIN_PASSIVE_FOR_MATCH,
            },
            "matched": False,
            "name": None,
            "score": None,
            "threshold": None,
            "margin": None,
        }

    result = verify_user_from_frame(frame)
    result["liveness"] = {
        "passed": True,
        "anti_spoof_message": anti_spoof_msg,
        "passive_score": round(passive_score, 4),
        "minimum_passive_score": MIN_PASSIVE_FOR_MATCH,
    }
    return result


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


def verify_realtime_result(camera_index: int = 0, show_window: bool = False) -> dict[str, Any]:
    """Run webcam liveness and then perform real-time identity matching.

    Why we need this:
    - This is the most practical flow for login-like scenarios.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return {
            "success": False,
            "status": "camera_unavailable",
            "message": "Camera unavailable",
            "matched": False,
            "name": None,
        }

    reset_liveness_state()
    last_msg = None
    start_time = time.time()
    liveness_message = "Do blink and slight head movement"

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            if show_window:
                cv2.destroyAllWindows()
            return {
                "success": False,
                "status": "camera_read_failed",
                "message": "Camera read failed",
                "matched": False,
                "name": None,
            }

        is_live, msg = check_liveness(frame)
        LOGGER.debug("Liveness status: %s", msg)
        liveness_message = msg

        if msg != last_msg:
            last_msg = msg

        if is_live:
            break

        if time.time() - start_time > LIVENESS_TIMEOUT_SECONDS:
            cap.release()
            if show_window:
                cv2.destroyAllWindows()
            return {
                "success": False,
                "status": "liveness_timeout",
                "message": "Liveness timeout, please try again",
                "liveness": {"passed": False, "message": liveness_message},
                "matched": False,
                "name": None,
            }

        if show_window:
            cv2.imshow("Liveness Check", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return {
                    "success": False,
                    "status": "cancelled",
                    "message": "Cancelled",
                    "matched": False,
                    "name": None,
                }

    LOGGER.info("Liveness passed. Matching identity from live frames.")

    # After challenge steps (blink/turn), collect a short burst of frames and
    # try matching each one. This avoids matching on a side-pose frame.
    matched_result: dict[str, Any] | None = None
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

        result = _match_embedding_result(embedding)
        if result["status"] == "matched":
            matched_result = result
            break

    cap.release()
    if show_window:
        cv2.destroyAllWindows()

    if matched_result is not None:
        LOGGER.info("Realtime verification success: %s", matched_result)
        matched_result["liveness"] = {"passed": True, "message": liveness_message}
        return matched_result

    LOGGER.info("Realtime verification result: Unknown person")
    return {
        "success": True,
        "status": "unknown",
        "message": "Unknown person",
        "liveness": {"passed": True, "message": liveness_message},
        "matched": False,
        "name": None,
    }


def verify_realtime(camera_index: int = 0) -> str:
    """Return the legacy CLI message for realtime verification."""
    print("Checking liveness... Do blink and slight head movement")
    result = verify_realtime_result(camera_index=camera_index, show_window=True)
    return result["message"]
