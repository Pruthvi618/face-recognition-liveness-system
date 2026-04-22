# FaceRecognition Project

Production-oriented face recognition service with registration, verification, and realtime liveness checks.

## Features
- DeepFace ArcFace embeddings for identity matching
- PostgreSQL-backed enrollment store with pooled DB connections
- Realtime liveness: blink + head movement + anti-spoof screen artifact checks
- Environment-driven configuration (no hardcoded secrets required)
- CLI entrypoints for register, verify, liveness-test, and realtime verification

## Project Structure
- `app/`: core modules (`model`, `service`, `database`, `liveness`, `utils`, `logger`)
- `scripts/`: runnable CLI scripts
- `images/`: sample images for local testing
- `config.py`: validated environment-based settings
- `.env.example`: template for local runtime configuration

## Prerequisites
- Python 3.11+
- PostgreSQL 14+
- Webcam (for realtime/liveness scripts)

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create env file from template:
   ```bash
   copy .env.example .env
   ```
4. Update `.env` with your PostgreSQL credentials.
5. Ensure PostgreSQL is running. Tables are auto-created on first app use.

## CLI Usage

Register user:
```bash
python scripts/register_user.py --name "Pruthvi" --image "images/test2.jpg"
```

Verify from image:
```bash
python scripts/verify_user.py --image "images/test.jpg"
```

Run standalone blink test:
```bash
python scripts/liveness_test.py --camera-index 0
```

Realtime verify (liveness + match):
```bash
python scripts/realtime_verify.py --camera-index 0
```

## Environment Variables
Configured via `.env` or system environment:
- `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`
- `DB_MIN_CONNECTIONS`, `DB_MAX_CONNECTIONS`
- `MODEL_NAME`, `MODEL_DETECTOR_BACKEND`
- `LIVENESS_TIMEOUT_SECONDS`, `MIN_PASSIVE_FOR_MATCH`
- `LOG_LEVEL`

## Production Notes
- Use strong database passwords and secret management (vault/KMS) in production.
- Run behind a service layer/API with authentication and rate limiting.
- Add monitoring around liveness failures, spoof detections, and match confidence.
- For high-security environments, pair heuristic anti-spoofing with a dedicated anti-spoof model.
