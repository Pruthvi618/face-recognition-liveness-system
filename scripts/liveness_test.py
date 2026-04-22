"""CLI entrypoint for standalone blink/liveness testing.

Why this file exists:
- Lets you validate camera access and liveness logic without full face matching.
"""

import argparse
import cv2
from pathlib import Path
import sys

# Add project root so `app.*` imports work when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.logger import configure_logging
from app.liveness import detect_blink


def main() -> None:
    """Run standalone blink detection loop."""
    parser = argparse.ArgumentParser(description="Run standalone blink/liveness test.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    args = parser.parse_args()

    configure_logging()
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print("Camera unavailable")
        return

    print("Blink to verify liveness...")

    while True:
        # Read one frame at a time and stop if camera read fails.
        ret, frame = cap.read()
        if not ret:
            break

        # Successful blink means we treat the user as live and end this test.
        if detect_blink(frame):
            print("Liveness confirmed!")
            break

        cv2.imshow("Liveness Check", frame)

        # Allow manual exit with q.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Always release camera resources and close OpenCV windows.
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
