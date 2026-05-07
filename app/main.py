"""FastAPI entrypoint for face registration and verification.

Why this file exists:
- Expose the registration and verification flows as HTTP APIs.
- Keep request parsing, validation, and response formatting outside the core
  service layer.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import argparse
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Add project root so `app.*` imports work when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.database import close_pool, ensure_schema
from app.logger import configure_logging, get_logger
from app.service import (
    register_user_from_frame,
    verify_frame_with_liveness,
    verify_realtime_result,
    verify_user_from_frame,
)

LOGGER = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared app resources."""
    configure_logging()
    ensure_schema()
    LOGGER.info("FastAPI face recognition service started")
    try:
        yield
    finally:
        close_pool()


api = FastAPI(
    title="Face Recognition API",
    description="Register users and verify faces with DeepFace embeddings and liveness checks.",
    version="1.0.0",
    lifespan=lifespan,
)

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _read_image_upload(image: UploadFile) -> Any:
    """Decode an uploaded image into an OpenCV BGR frame."""
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image file")

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Image file is empty")

    buffer = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    return frame


@api.get("/")
def root() -> dict[str, Any]:
    """Basic API discovery response."""
    return {
        "service": "Face Recognition API",
        "docs": "/docs",
        "endpoints": {
            "register": "POST /register",
            "verify_image": "POST /verify",
            "verify_realtime": "POST /verify/realtime",
            "health": "GET /health",
        },
    }


@api.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@api.post("/register")
async def register_user(
    name: str = Form(..., min_length=1, description="Person name to register"),
    image: UploadFile = File(..., description="Face image for enrollment"),
) -> dict[str, Any]:
    """Register a user from a form field and uploaded face image."""
    frame = await _read_image_upload(image)
    result = register_user_from_frame(name=name.strip(), frame=frame)
    if not result["success"]:
        raise HTTPException(status_code=422, detail=result)
    return result


@api.post("/verify")
async def verify_user(
    image: UploadFile = File(..., description="Face image to verify"),
    require_liveness: bool = Form(False, description="Run passive image liveness checks first"),
) -> dict[str, Any]:
    """Verify a user from an uploaded image and return a JSON decision."""
    frame = await _read_image_upload(image)
    if require_liveness:
        result = verify_frame_with_liveness(frame)
    else:
        result = verify_user_from_frame(frame)

    if result["status"] in {"no_face_detected", "liveness_failed"}:
        raise HTTPException(status_code=422, detail=result)
    return result


@api.post("/verify/realtime")
def verify_realtime(
    camera_index: int = Form(0, description="OpenCV camera index on the API server machine"),
) -> dict[str, Any]:
    """Run realtime webcam liveness on the server and return the match decision."""
    result = verify_realtime_result(camera_index=camera_index, show_window=False)
    if not result["success"]:
        raise HTTPException(status_code=422, detail=result)
    return result


def _run_from_cli() -> None:
    """Allow running the API via `python app/main.py`."""
    parser = argparse.ArgumentParser(description="Run Face Recognition FastAPI server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run("app.main:api", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    _run_from_cli()
