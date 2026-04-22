"""CLI entrypoint for still-image user verification.

Why this file exists:
- Provides a simple local test for matching a face image to registered users.
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
    """Parse CLI args and run image verification."""
    parser = argparse.ArgumentParser(description="Verify a user from an image.")
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()

    configure_logging()
    from app.service import verify_user

    print(verify_user(args.image))


if __name__ == "__main__":
    main()
