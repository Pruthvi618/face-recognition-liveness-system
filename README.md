# FaceRecognition Project

Production-oriented FastAPI face recognition service with registration, verification, and realtime liveness checks.

## Features
- DeepFace ArcFace embeddings for identity matching
- PostgreSQL-backed enrollment store with pooled DB connections
- Realtime liveness: blink + head movement + anti-spoof screen artifact checks
- FastAPI endpoints for enrollment, still-image verification, and realtime webcam verification
- Environment-driven configuration (no hardcoded secrets required)
- CLI entrypoints for register, verify, liveness-test, and realtime verification

## Project Structure
- `app/`: API and core modules (`main`, `model`, `service`, `database`, `liveness`, `utils`, `logger`)
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

## FastAPI Usage

Start the API server:
```bash
uvicorn app.main:api --reload
```

Or run directly with Python:
```bash
python app/main.py --reload --port 8000
```

Open the interactive API form:
```text
http://127.0.0.1:8000/docs
```

If `8000` is already used, choose another port:
```bash
python app/main.py --reload --port 8011
```

Register user:
```bash
curl -X POST "http://127.0.0.1:8000/register" ^
  -F "name=Pruthvi" ^
  -F "image=@images/test2.jpg"
```

Verify from image:
```bash
curl -X POST "http://127.0.0.1:8000/verify" ^
  -F "image=@images/test.jpg"
```

Verify from image with passive liveness:
```bash
curl -X POST "http://127.0.0.1:8000/verify" ^
  -F "require_liveness=true" ^
  -F "image=@images/test.jpg"
```

Realtime verify using the API server machine camera:
```bash
curl -X POST "http://127.0.0.1:8000/verify/realtime" ^
  -F "camera_index=0"
```

Example verification response:
```json
{
  "success": true,
  "status": "matched",
  "message": "Matched: Pruthvi (score=0.78)",
  "matched": true,
  "name": "Pruthvi",
  "score": 0.7812,
  "threshold": 0.65,
  "margin": 0.12
}
```

## CLI Usage

Register user interactively:
```bash
python scripts/register_user.py
```

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
- Add authentication and rate limiting before exposing the API outside a trusted network.
- Add monitoring around liveness failures, spoof detections, and match confidence.
- For high-security environments, pair heuristic anti-spoofing with a dedicated anti-spoof model.
