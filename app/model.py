"""Model wrapper around DeepFace embedding generation.

Why this file exists:
- Keep all model/backend choices in one place.
- Expose one simple function for the rest of the app to request embeddings.
"""

from __future__ import annotations

from typing import Any

from deepface import DeepFace

from app.logger import get_logger
from config import settings

LOGGER = get_logger(__name__)


def get_embedding(image_input: Any, enforce_detection: bool = True) -> list[float] | None:
    """
    Accepts an image path or a numpy frame (BGR) and returns an ArcFace embedding.

    Why we need this:
    - Registration and verification both depend on the same embedding pipeline.
    - Centralizing this prevents drift between different script entrypoints.
    """
    try:
        embedding = DeepFace.represent(
            img_path=image_input,
            model_name=settings.model_name,
            detector_backend=settings.model_detector_backend,
            enforce_detection=enforce_detection,
            align=True,
        )
        return embedding[0]["embedding"]
    except Exception:
        LOGGER.exception("Failed to extract embedding")
        return None


if __name__ == "__main__":
    emb = get_embedding("images/test2.jpg")
    
    if emb:
        print("Embedding length:", len(emb))
    else:
        print("No face detected")
