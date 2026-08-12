# 🎙️ AcousticSpace — AI Deepfake Audio Detection

[![AcousticSpace CI](https://github.com/Shreyasi-15/ds-ml-internship/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Shreyasi-15/ds-ml-internship/actions/workflows/ci.yml)

AcousticSpace is an end-to-end research system that classifies uploaded audio as **bonafide human speech** or **AI-generated spoof audio**. The active classifier is a domain-adapted **Wav2Vec2** model, supported by acoustic measurements, breathing-band evidence, waveform playback, and suspicious four-second segment analysis.

> AcousticSpace is a research prototype and must not be treated as a final forensic verdict.

## Overview

The system provides:

- domain-adapted Wav2Vec2 bonafide/spoof classification
- FastAPI inference and feature-extraction endpoints
- React and TypeScript analyst dashboard
- interactive waveform and suspicious-segment evidence
- recording probabilities and model confidence
- reverb, breathing-band, RMS, spectral-centroid, and spectral-bandwidth evidence
- authenticated SQLite-backed per-user history
- 30-day retention for unkept analyses
- permanent saved history with Keep and Delete controls
- Docker Compose deployment and GitHub Actions validation

## How It Works

1. The analyst uploads an audio recording.
2. The backend validates and decodes mono audio at 16 kHz.
3. Wav2Vec2 produces bonafide and spoof probabilities for the recording.
4. Four-second windows provide supporting suspicious-segment evidence.
5. Librosa calculates independent acoustic measurements.
6. The dashboard presents the prediction, confidence, waveform, metadata, evidence, and history controls.

```text
Audio upload
    ↓
Validation and 16 kHz preprocessing
    ↓
Domain-adapted Wav2Vec2 → Final prediction
    ↓
Four-second windows → Segment evidence
    ↓
Acoustic evidence, dashboard, and retained history
```

## Final Evaluation

### Balanced ASVspoof evaluation

| Metric | Result |
|---|---:|
| Evaluated recordings | 1,000 |
| Accuracy | 99.0% |
| Precision | 99.40% |
| Recall | 98.60% |
| F1 score | 99.00% |
| Equal Error Rate | 0.90% |

Confusion matrix: `[[497, 3], [7, 493]]`

### Private domain test

| Metric | Result |
|---|---:|
| Evaluated recordings | 10 |
| Accuracy | 100% |
| Precision | 100% |
| Recall | 100% |
| F1 score | 100% |
| Equal Error Rate | 0% |

Confusion matrix: `[[5, 0], [0, 5]]`

The domain test is extremely small. Its perfect result is preliminary and does not establish real-world generalization.

## Technology Stack

| Component | Technology |
|---|---|
| Classifier | PyTorch and HuggingFace Wav2Vec2 |
| Audio processing | Python and Librosa |
| API | FastAPI and Uvicorn |
| Dashboard | React and TypeScript |
| History | SQLite |
| Waveform | WaveSurfer |
| Deployment | Docker Compose and Nginx |
| CI/CD | GitHub Actions |
| Primary dataset | ASVspoof 2019 Logical Access |

## Run the Project

The trained checkpoint is intentionally excluded from Git. Place these files under:

```text
AcousticSpace/backend/models/wav2vec2_asvspoof_domain_v2/best
```

Required files:

```text
config.json
model.safetensors
preprocessor_config.json
training_args.bin
```

With Docker Desktop running:

```powershell
cd AcousticSpace
docker compose up -d --build
docker compose ps
```

- Dashboard: `http://localhost:8080`
- Backend API: `http://localhost:8000`
- API documentation: `http://localhost:8000/docs`

For local development, run the FastAPI backend on port `8000` and the Vite frontend on port `5173`.

## Project Components

```text
AcousticSpace/
├── backend/                 # FastAPI, Wav2Vec2 inference, training, and evaluation
├── dataset/                 # Manifests and ignored private domain-audio structure
├── docs/                    # Weekly and final reports
├── frontend/                # React analyst dashboard
├── scripts/                 # Dataset and evaluation utilities
└── docker-compose.yml       # Backend and frontend deployment
```

## Validation

- 14 backend tests passed
- frontend ESLint passed
- frontend production build passed
- Python modules compiled successfully
- local bonafide and spoof inference tests passed
- model checkpoints and private recordings are excluded from Git

## Documentation

- [Complete AcousticSpace documentation](AcousticSpace/README.md)
- [Final project report](AcousticSpace/docs/final_project_report.md)
- [Historical Week 4 AST results](AcousticSpace/docs/week4_results.md)
- [Wav2Vec2 ASVspoof metrics](AcousticSpace/backend/artifacts/wav2vec2_asv_eval_metrics.json)
- [Wav2Vec2 domain-test metrics](AcousticSpace/backend/artifacts/wav2vec2_domain_test_metrics.json)

## Limitations

- The 10-record domain test is too small for a general performance claim.
- Performance may change across languages, rooms, microphones, codecs, and generation systems.
- Acoustic measurements are supporting analyst evidence; they are not explicitly fused into the Wav2Vec2 classifier.
- Breathing cadence and RT60 are transparent signal proxies rather than validated forensic measurements.
- The separately distributed checkpoint is required for inference.
- The system has not undergone forensic, adversarial, bias, or production-scale validation.

## Author

**Shreyasi Chowdhury**  
Data Science & Machine Learning Intern  
Infotact Solutions
