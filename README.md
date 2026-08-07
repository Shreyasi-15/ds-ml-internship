# 🎙️ AcousticSpace — AI Deepfake Audio Detection

[![AcousticSpace CI](https://github.com/Shreyasi-15/ds-ml-internship/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Shreyasi-15/ds-ml-internship/actions/workflows/ci.yml)

AcousticSpace is an AI-powered system that checks whether an uploaded audio recording is **bonafide human speech** or **AI-generated spoof audio**.

It uses a fine-tuned **Audio Spectrogram Transformer (AST)** together with supporting acoustic evidence such as reverberation, breathing-band energy, waveform patterns, and suspicious four-second segments.

---

## 📌 Overview

Modern AI-generated voices can sound highly realistic. AcousticSpace helps an analyst examine a recording through an accessible dashboard.

The system provides:

- Fine-tuned Audio Spectrogram Transformer
- FastAPI prediction backend
- React and TypeScript dashboard
- Interactive audio waveform
- Recording-level confidence scores
- Suspicious segment detection
- Acoustic and breathing-cadence evidence
- Browser-based analysis history
- Docker deployment
- GitHub Actions validation

---

## 🧠 How It Works

1. The user uploads an audio recording.
2. The system validates it and loads mono audio at 16 kHz.
3. The trained AST model examines the complete recording for the final classification.
4. The recording is separately divided into four-second segments for suspicious-segment evidence.
5. Acoustic and breathing-cadence measurements are calculated as supporting evidence.
6. The dashboard displays the final prediction, probabilities, waveform, segment evidence, and acoustic measurements.

```text
Audio Upload
     ↓
Validation and Preprocessing
     ↓
Complete Recording → AST → Final Prediction
     ↓
Four-Second Analysis → Suspicious Segment Evidence
     ↓
Acoustic Evidence and React Dashboard
```

---

## 📊 Final Evaluation

Evaluation used a deterministic balanced subset of **4,000 unseen recordings** from the official ASVspoof 2019 LA evaluation split.

| Metric | Result |
|---|---:|
| Accuracy | 92.63% |
| Precision | 98.97% |
| Recall | 86.15% |
| F1 score | 92.11% |
| Equal Error Rate | 4.78% |
| Evaluated recordings | 4,000 |

### Confusion Matrix

| Actual / Predicted | Bonafide | Spoof |
|---|---:|---:|
| Bonafide | 1,982 | 18 |
| Spoof | 277 | 1,723 |

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| AI model | PyTorch and HuggingFace AST |
| Audio processing | Python and Librosa |
| Backend | FastAPI and Uvicorn |
| Frontend | React and TypeScript |
| Waveform | WaveSurfer |
| Deployment | Docker Compose and Nginx |
| CI/CD | GitHub Actions |
| Dataset | ASVspoof 2019 Logical Access |

---

## 🚀 Run the Project

Place the trained AST checkpoint under:

```text
AcousticSpace/backend/models/ast_asvspoof_v1/best
```

Required checkpoint files:

```text
config.json
model.safetensors
preprocessor_config.json
training_args.bin
```

Start Docker Desktop, then run:

```powershell
cd AcousticSpace
docker compose up -d
docker compose ps
```

Open:

- Dashboard: `http://localhost:8080`
- Backend API: `http://localhost:8000`
- API documentation: `http://localhost:8000/docs`

---

## 📁 Main Project Components

```text
AcousticSpace/
├── backend/                 # FastAPI, AST inference and evaluation
├── dataset/                 # Dataset documentation and manifests
├── docs/                    # Weekly and final reports
├── frontend/                # React analyst dashboard
├── scripts/                 # Dataset and evaluation utilities
└── docker-compose.yml       # Application deployment
```

---

## ✅ Validation

- 14 backend tests passed
- Frontend ESLint passed
- Frontend production build passed
- npm audit reported zero vulnerabilities
- Docker backend and frontend were healthy
- Real AST prediction endpoint verified
- GitHub Actions checks passed
- Model binaries excluded from Git

---

## 📚 Documentation

- [Complete AcousticSpace documentation](AcousticSpace/README.md)
- [Final project report](AcousticSpace/docs/final_project_report.md)
- [Week 4 results](AcousticSpace/docs/week4_results.md)
- [Final evaluation metrics](AcousticSpace/backend/artifacts/ast_eval_metrics.json)
- [Final confusion matrix](AcousticSpace/backend/artifacts/ast_eval_confusion_matrix.png)

---

## ⚠️ Important Limitation

Acoustic features are currently displayed as supporting analyst evidence and are not directly fused into the AST model's prediction.

The final evaluation covered a deterministic balanced 4,000-record subset of the official unseen ASVspoof 2019 LA evaluation split, not the complete 71,237-record evaluation split.

AcousticSpace is a research prototype and must not be treated as a guaranteed forensic verdict.

## 👩‍💻 Author

**Shreyasi Chowdhury**<br>
Data Science & Machine Learning Intern<br>
Infotact Solutions