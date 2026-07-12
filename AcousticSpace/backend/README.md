# AcousticSpace — Backend (Week 1)

Week 1 scope only, per the project plan:
- Build FastAPI server
- Curate dataset (ASVspoof)
- Librosa pipeline to extract spectrograms + RIR features

No classifier yet — that's Week 2 (baseline CNN/Transformer model).

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run the server

```bash
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs — FastAPI gives you a free interactive
test page. Try the `/health` endpoint first, then `/extract-features`
by uploading any audio file.

## Run tests

```bash
pytest tests/ -v
```

Both tests should pass — they prove the pipeline correctly detects a
reverb tail (simulated room reflection) in a synthetic audio clip.

## Files

- `app/main.py` — FastAPI server, two endpoints: `/health` and `/extract-features`
- `app/audio_pipeline.py` — Librosa feature extraction (RT60, reverb ratio, breathing-band energy, mel-spectrogram)
- `app/dataset.py` — builds a manifest CSV from ASVspoof protocol files
- `tests/test_pipeline.py` — validates the pipeline on synthetic audio
