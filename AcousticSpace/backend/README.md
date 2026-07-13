# AcousticSpace backend — Week 1

FastAPI gateway, Librosa feature pipeline, ASVspoof-style dataset curation,
and automated tests. There is no classifier in Week 1.

## Setup and run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open `http://127.0.0.1:8000/docs` and try `/health` followed by
`/extract-features`.

## Test

```bash
python -m pytest
```

Tests cover feature shape/ranges, a dry-versus-reverberant synthetic signal,
dataset protocol validation, and API success/failure paths.

## Safety limits

- Allowed formats: WAV, FLAC, MP3, and M4A
- Maximum upload: 25 MB
- Maximum decoded duration: 300 seconds
- Temporary uploads are deleted after each request

The returned measurements are proxies. The API deliberately does not return a
deepfake label or probability before a Week 2 model is trained and evaluated.
