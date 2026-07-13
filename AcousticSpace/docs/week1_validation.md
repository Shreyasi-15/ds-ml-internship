# Week 1 validation record

## Scope

- FastAPI server and validated upload endpoint
- Librosa mel-spectrogram and acoustic proxy extraction
- ASVspoof-style manifest curation with label/file validation
- React + TypeScript upload dashboard
- Unit and API tests
- Redistributable synthetic smoke-test dataset

## Verification commands

From `AcousticSpace/backend`:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest
uvicorn app.main:app --reload --port 8000
```

From `AcousticSpace/frontend` in a second terminal:

```bash
npm install
copy .env.example .env
npm run build
npm run lint
npm run dev
```

Dataset smoke test from `AcousticSpace`:

```bash
python scripts/generate_demo_dataset.py
python backend/app/dataset.py --protocol dataset/demo/protocol.txt --audio-dir dataset/demo/audio --audio-ext .wav --output dataset/demo/manifest.csv --summary dataset/demo/dataset_summary.json --split demo --source-name "Synthetic smoke test"
```

## Acceptance checklist

- [x] `python -m pytest`: **8 passed** on Python 3.12.
- [x] `/health` returns HTTP 200 through the API test client.
- [x] A valid WAV returns a typed, 64-band mel feature response.
- [x] Empty and unsupported files return controlled 4xx errors.
- [x] The demo summary reports 4 files and 0 missing.
- [x] `npm run build` and `npm run lint` pass.
- [x] The frontend renders every typed response field and no real/fake verdict.

Automated verification was completed on 13 July 2026. Three non-failing
deprecation warnings originated inside Audioread's Python standard-library
imports; no project test failed.

The RT60, reverb, and breathing values are diagnostic proxies. They are not a
deepfake verdict and should not be presented as model accuracy.
