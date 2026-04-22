"""CLI entrypoint for webcam-based liveness + verification.

Why this file exists:
- Runs the full real-time flow from camera capture to identity decision.
"""

import argparse
from pathlib import Path
import sys

# Add project root so `app.*` imports work when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.logger import configure_logging


def main() -> None:
    """Parse CLI args and run realtime verification."""
    parser = argparse.ArgumentParser(description="Run realtime liveness and face verification.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    args = parser.parse_args()

    configure_logging()
    from app.service import verify_realtime

    print(verify_realtime(camera_index=args.camera_index))


if __name__ == "__main__":
    main()
