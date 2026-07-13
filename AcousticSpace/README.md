# AcousticSpace

Week 1 implementation for acoustic deepfake analysis using room/reverberation
proxies. The system accepts audio, extracts interpretable acoustic features,
and displays them in a React analyst dashboard.

## Week 1 status

**Implementation complete and verified: 8 backend tests, frontend build, and
frontend lint all pass.**

- FastAPI `/health` and `/extract-features` endpoints
- Upload validation, size limit, duration limit, and temporary-file cleanup
- Librosa mel-spectrogram, RT60 proxy, temporal-decay ratio, and quiet-frame
  breathing-band ratio
- ASVspoof-style protocol parser, manifest schema, label/file validation, and
  JSON dataset summaries
- Four-file synthetic smoke-test dataset (not training data)
- React + TypeScript dashboard configured through `VITE_API_BASE_URL`
- Pipeline, dataset, and API tests

The CNN/AST classifier and waveform visualization are **not started** because
they belong to Week 2.

## Architecture

| Module | Technology | Week 1 responsibility |
|---|---|---|
| Audio pipeline | Python, Librosa | Decode audio and calculate acoustic proxies |
| Dataset curation | Python, CSV/JSON | Validate protocols and build leakage-aware manifests |
| API gateway | FastAPI | Validate uploads and return a typed feature report |
| Analyst dashboard | React, TypeScript | Upload audio and display feature results |

## Project structure

```text
AcousticSpace/
├── backend/
│   ├── app/                 # API, schemas, configuration, pipeline, curation
│   ├── tests/               # pipeline, dataset, and endpoint tests
│   └── requirements.txt
├── dataset/
│   ├── README.md            # selected source and curation rules
│   └── demo/                # generated smoke-test WAVs and manifest
├── docs/week1_validation.md
├── frontend/                # React + TypeScript dashboard
└── scripts/generate_demo_dataset.py
```

## Quick start

Backend:

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest
uvicorn app.main:app --reload --port 8000
```

Frontend, in a second terminal:

```bash
cd frontend
npm install
copy .env.example .env
npm run dev
```

See `dataset/README.md` for curation commands and
`docs/week1_validation.md` for the submission checklist.

## Evidence boundary

An arbitrary speech clip does not contain a directly measured room impulse
response. AcousticSpace therefore reports RT60/reverb/breathing **proxies**.
They may become model inputs in Week 2, but they are not proof that audio is
real or fake.

## Author

Shreyasi — Data Science & Machine Learning Intern, Infotact Solutions
