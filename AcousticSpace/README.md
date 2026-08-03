# AcousticSpace

AcousticSpace is an end-to-end deepfake-audio research system built with a fine-tuned Audio Spectrogram Transformer, FastAPI, React, Librosa, Docker, and GitHub Actions.

The application classifies uploaded recordings as bonafide or spoof, displays recording and segment probabilities, renders an interactive waveform, and reports independent acoustic evidence for analyst review.

> AcousticSpace is a research prototype, not a final forensic decision system.

## Final Project Status

The planned implementation is complete and validated:

- Fine-tuned HuggingFace Audio Spectrogram Transformer.
- Balanced AST training on ASVspoof 2019 LA.
- Reproducible evaluation on unseen evaluation recordings.
- FastAPI AST inference integration.
- Four-second suspicious-segment analysis.
- Acoustic and breathing-cadence diagnostics.
- React and TypeScript analyst dashboard.
- Browser-based analysis history.
- Docker Compose deployment.
- GitHub Actions continuous integration.
- 13 backend tests passing.
- Frontend lint and production build passing.
- Backend and frontend containers healthy.

## Final Evaluation Results

The final evaluation used a deterministic balanced 4,000-record subset of the official unseen ASVspoof 2019 LA evaluation split.

| Metric | Result |
|---|---:|
| Evaluated recordings | 4,000 |
| Bonafide recordings | 2,000 |
| Spoof recordings | 2,000 |
| Accuracy | 92.625% |
| Precision | 98.966% |
| Recall | 86.150% |
| F1 score | 92.114% |
| Equal Error Rate | 4.775% |
| GPU evaluation time | 398.12 seconds |
| Throughput | 10.05 recordings/second |

Confusion matrix:

| Actual / Predicted | Bonafide | Spoof |
|---|---:|---:|
| Bonafide | 1,982 | 18 |
| Spoof | 277 | 1,723 |

The model has high spoof precision but missed 277 of the 2,000 spoof recordings. Results must therefore be interpreted as decision support rather than a forensic verdict.

The evaluation covers a balanced subset, not the complete 71,237-record evaluation split.

Detailed methodology and limitations are documented in [docs/final_project_report.md](docs/final_project_report.md).

## Architecture

| Module | Technology | Responsibility |
|---|---|---|
| Audio pipeline | Python, Librosa | Audio decoding, resampling, segmentation, and acoustic diagnostics |
| Transformer classifier | PyTorch, HuggingFace AST | Bonafide/spoof classification |
| API gateway | FastAPI, Uvicorn | Upload validation, feature extraction, and AST inference |
| Analyst dashboard | React, TypeScript | Upload, waveform, prediction, segment evidence, and history |
| Reverse proxy | Nginx | Frontend hosting and `/api` proxy |
| Deployment | Docker Compose | Reproducible backend and frontend services |
| Continuous integration | GitHub Actions | Backend, frontend, and repository-safety checks |

## Model

The production inference model is:

`ast-asvspoof2019-la-v1`

It was initialized from:

`MIT/ast-finetuned-audioset-10-10-0.4593`

Training configuration:

| Setting | Value |
|---|---|
| Training recordings | 5,000 balanced |
| Development recordings | 4,000 balanced |
| Epochs | 3 |
| Batch size | 4 |
| Learning rate | 2e-5 |
| Sampling rate | 16 kHz |
| Hardware | NVIDIA Tesla T4 |
| Seed | 42 |

Development metrics were:

- Accuracy: 99.275%
- Precision: 98.715%
- Recall: 99.850%
- F1: 99.279%
- EER: 0.600%

Development metrics were used during model selection and are not final unseen results.

## Model Checkpoint Setup

The trained checkpoint is approximately 329 MB and is intentionally excluded from Git.

Place the checkpoint files in:

```text
AcousticSpace/backend/models/ast_asvspoof_v1/best/