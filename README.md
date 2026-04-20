# FaceRecognition Project

A Python-based face recognition system with user registration, verification, and liveness checks.

## Features
- Register users from face images
- Verify users using face embeddings
- Realtime verification flow with liveness gating
- PostgreSQL-backed embedding storage

## Project Structure
- `app/` core logic (model, service, database, liveness, utils)
- `scripts/` runnable entry scripts for register/verify/liveness/realtime
- `images/` sample input images

## Setup
1. Create and activate virtual environment
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure PostgreSQL connection in `app/database.py`

## Run
- Register:
  ```bash
  python scripts/register_user.py
  ```
- Verify (image):
  ```bash
  python scripts/verify_user.py
  ```
- Liveness test:
  ```bash
  python scripts/liveness_test.py
  ```
- Realtime verify:
  ```bash
  python scripts/realtime_verify.py
  ```

## Notes
- This repo currently uses local DB credentials in code. Move secrets to environment variables before publishing publicly.