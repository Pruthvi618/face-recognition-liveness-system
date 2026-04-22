"""CLI entrypoint for user registration.

Why this file exists:
- Gives a quick command-line way to enroll a person into the database.
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
    """Parse CLI args and run registration."""
    parser = argparse.ArgumentParser(description="Register a user embedding from an image.")
    parser.add_argument("--name", required=True, help="Person name to register")
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()

    configure_logging()
    from app.service import register_user

    print(register_user(args.name, args.image))


if __name__ == "__main__":
    main()
