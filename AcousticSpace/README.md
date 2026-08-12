# AcousticSpace

AcousticSpace is a research-oriented deepfake-audio analysis application built with a domain-adapted HuggingFace Wav2Vec2 classifier, FastAPI, React, TypeScript, Librosa, SQLite, Docker, and GitHub Actions.

The application classifies recordings as bonafide or spoof, reports recording and four-second segment probabilities, renders an interactive waveform, calculates supporting acoustic evidence, and maintains authenticated per-user history.

> The output is decision-support evidence, not a final forensic verdict.

## Final Project Status

Completed functionality includes:

- balanced Wav2Vec2 training and evaluation utilities
- private domain-adaptation dataset validation and bundling
- domain-adapted Wav2Vec2 inference in FastAPI
- recording-level bonafide and spoof probabilities
- suspicious four-second segment evidence
- audio metadata and acoustic-feature reporting
- interactive React waveform and probability chart
- authenticated SQLite-backed analysis history
- automatic deletion of unkept records after 30 days
- persistent saved history with explicit Keep and Delete actions
- Docker Compose deployment using the Wav2Vec2 checkpoint
- GitHub Actions backend, frontend, and repository-safety checks

## Architecture

| Module | Technology | Responsibility |
|---|---|---|
| Audio pipeline | Python, Librosa | Decode, resample, segment, and calculate acoustic evidence |
| Classifier | PyTorch, HuggingFace Wav2Vec2 | Bonafide/spoof probabilities |
| API gateway | FastAPI, Uvicorn | Authentication, upload validation, inference, features, history |
| History | SQLite | Per-user recent and saved analyses |
| Dashboard | React, TypeScript | Upload, waveform, prediction, evidence, statistics, history |
| Reverse proxy | Nginx | Frontend hosting and API proxy |
| Deployment | Docker Compose | Backend and frontend services |
| CI | GitHub Actions | Tests, build, compilation, and repository safety |

## Inference Workflow

1. The frontend accepts WAV, MP3, MPEG, FLAC, or M4A audio.
2. FastAPI validates the file and Librosa decodes mono audio at 16 kHz.
3. The Wav2Vec2 feature extractor prepares the waveform.
4. `Wav2Vec2ForSequenceClassification` returns bonafide and spoof logits.
5. Softmax probabilities provide the recording result and confidence.
6. Four-second windows are evaluated separately as segment evidence.
7. Acoustic features are calculated independently and displayed to the analyst.
8. The completed prediction is stored under the authenticated user.

The active model version is:

```text
wav2vec2-asvspoof-domain-v2
```

## Acoustic Evidence

The dashboard reports:

- RT60 proxy when measurable
- reverb ratio
- breathing-band ratio
- duration and peak amplitude
- RMS energy
- spectral centroid
- spectral bandwidth
- mel-spectrogram shape

These values change with the analyzed recording. They support interpretation but are not directly fused into the classifier's decision.

## History Retention

- New analyses enter **Recent server-saved analyses**.
- Unkept recent analyses expire automatically after 30 days.
- **Keep** moves a record into Saved History.
- Saved records remain until the user selects **Delete**.
- Every history query and mutation is restricted to the authenticated user.

## Dataset and Domain Adaptation

The base bundle contains balanced ASVspoof 2019 Logical Access training, development, and evaluation manifests. Private domain recordings are organized by split and label:

```text
dataset/domain_adaptation/
├── train/bonafide
├── train/spoof
├── test/bonafide
└── test/spoof
```

`prepare_domain_adaptation.py` validates class counts and rejects identical train/test audio. `prepare_mixed_wav2vec2_bundle.py` mixes the private training recordings into the base bundle while keeping the private domain test separate.

Private recordings and generated bundles remain outside Git.

## Training

Training used the mixed ASVspoof and private domain-adaptation bundle while keeping the private domain test isolated. The Colab run trained Wav2Vec2 for three epochs on a Tesla T4 and saved the best checkpoint under `best/`.

The repository contains:

- `backend/ml/wav2vec2_data.py`
- `backend/ml/train_wav2vec2.py`
- `backend/ml/evaluate_wav2vec2.py`
- `backend/ml/prepare_domain_adaptation.py`
- `backend/ml/prepare_mixed_wav2vec2_bundle.py`

## Evaluation Results

### Balanced ASVspoof evaluation

| Metric | Result |
|---|---:|
| Evaluated recordings | 1,000 |
| Accuracy | 99.0% |
| Precision | 99.395% |
| Recall | 98.6% |
| F1 | 98.996% |
| EER | 0.9% |

Confusion matrix: `[[497, 3], [7, 493]]`

### Private domain test

| Metric | Result |
|---|---:|
| Evaluated recordings | 10 |
| Accuracy | 100% |
| Precision | 100% |
| Recall | 100% |
| F1 | 100% |
| EER | 0% |

Confusion matrix: `[[5, 0], [0, 5]]`

The domain test result is preliminary because it contains only five recordings per class.

## Checkpoint Setup

The checkpoint is intentionally excluded from Git. Extract or copy the trained files to:

```text
backend/models/wav2vec2_asvspoof_domain_v2/best
```

Required files:

- `config.json`
- `model.safetensors`
- `preprocessor_config.json`
- `training_args.bin`

The model was saved using Transformers 5.13.1. Use the pinned backend requirements to avoid incompatible weight-parametrization warnings.

## Local Development

Backend:

```powershell
$backend = (Resolve-Path .\backend).Path
& "$backend.venv\Scripts\python.exe" -m uvicorn app.main:app --reload --port 8000
```

Frontend, in a separate terminal:

```powershell
Set-Location .\frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Docker Deployment

With Docker Desktop running and the checkpoint installed:

```powershell
docker compose up -d --build
docker compose ps
docker compose logs backend --tail 50
```

Open `http://localhost:8080`.

Docker mounts `backend/models` read-only and sets:

```text
ACOUSTICSPACE_MODEL_PATH=/app/models/wav2vec2_asvspoof_domain_v2/best
```

## Validation

- backend Pytest suite: 14 tests passed
- frontend ESLint: passed
- frontend production build: passed
- Python compilation: passed
- local Wav2Vec2 checkpoint loading: passed
- genuine and spoof local inference smoke tests: passed
- private audio and model binaries: excluded from Git

## Limitations

1. The private domain test has only 10 recordings and cannot establish real-world generalization.
2. Results may not transfer to new datasets, generation systems, languages, codecs, rooms, microphones, or noise conditions.
3. Acoustic and breathing evidence is displayed independently rather than fused into Wav2Vec2.
4. Suspicious four-second windows are supporting evidence rather than an independent final verdict.
5. RT60 and breathing cadence are signal proxies, not validated forensic measurements.
6. CPU inference can be slow for long recordings.
7. The checkpoint must be distributed separately.
8. The system has not undergone production security, adversarial, bias, or forensic validation.

## Historical Work

Earlier project phases implemented a baseline CNN and then an Audio Spectrogram Transformer. Those artifacts remain for reproducibility, but the active application model is the domain-adapted Wav2Vec2 checkpoint.
