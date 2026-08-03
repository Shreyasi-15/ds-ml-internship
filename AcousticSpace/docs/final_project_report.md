# AcousticSpace Final Project Report

## 1. Project Overview

AcousticSpace is a research-oriented deepfake-audio analysis system. It combines a fine-tuned Audio Spectrogram Transformer (AST) classifier with independently reported acoustic evidence, including reverb, waveform, low-frequency breathing-cadence, and suspicious-segment diagnostics.

The system consists of:

- A Python and Librosa audio-processing pipeline.
- A HuggingFace Audio Spectrogram Transformer classifier.
- A FastAPI inference gateway.
- A React and TypeScript analyst dashboard.
- Docker Compose deployment.
- GitHub Actions continuous integration.

The detector is intended for research and analyst review. Its output must not be treated as a final forensic verdict.

## 2. System Architecture

Uploaded audio passes through the following workflow:

1. The frontend validates the selected audio file.
2. Nginx forwards API requests to the FastAPI backend.
3. Librosa decodes the recording as mono audio at 16 kHz.
4. The recording is divided into four-second segments.
5. The AST feature extractor creates model inputs.
6. The trained AST checkpoint classifies each segment.
7. Segment probabilities are aggregated into a recording-level result.
8. Acoustic features and breathing-cadence diagnostics are calculated separately.
9. The dashboard displays the prediction, confidence, segment evidence, waveform, and acoustic diagnostics.

The AST classifier consumes spectrogram-derived inputs. RIR, reverb, and breathing diagnostics are currently presented as independent analyst evidence and are not explicitly fused into the AST model's decision.

## 3. Dataset

The project uses the ASVspoof 2019 Logical Access dataset.

The official protocol files were converted into CSV manifests containing:

- `filepath`
- `label`
- `label_name`
- `speaker_id`
- `attack_id`
- `split`

Label mapping:

- `0`: bonafide
- `1`: spoof

Training and development data were sampled reproducibly and balanced by class to reduce the effect of the original class imbalance.

## 4. AST Training Configuration

| Setting | Value |
|---|---|
| Base model | `MIT/ast-finetuned-audioset-10-10-0.4593` |
| Training recordings | 5,000 |
| Training balance | 2,500 bonafide and 2,500 spoof |
| Development recordings | 4,000 |
| Development balance | 2,000 bonafide and 2,000 spoof |
| Epochs | 3 |
| Batch size | 4 |
| Learning rate | 2e-5 |
| Sampling rate | 16 kHz |
| Training hardware | NVIDIA Tesla T4 |
| Random seed | 42 |
| Saved model version | `ast-asvspoof2019-la-v1` |

The trained model checkpoint is approximately 329 MB and is excluded from Git. It must be installed separately under:

`AcousticSpace/backend/models/ast_asvspoof_v1/best`

Required checkpoint files:

- `config.json`
- `model.safetensors`
- `preprocessor_config.json`
- `training_args.bin`

## 5. Development Results

The following metrics were measured on the balanced 4,000-record development subset:

| Metric | Result |
|---|---:|
| Accuracy | 99.275% |
| Precision | 98.715% |
| Recall | 99.850% |
| F1 score | 99.279% |
| Equal Error Rate | 0.600% |
| Evaluation loss | 0.0483 |

Development results were used during model selection and therefore are not treated as final unseen performance.

## 6. Unseen Evaluation Methodology

Final evaluation used a deterministic balanced 4,000-record subset of the official unseen ASVspoof 2019 LA evaluation split.

The subset contained:

- 2,000 bonafide recordings.
- 2,000 spoof recordings.
- Random seed 42.
- No missing referenced audio files.

The source evaluation manifest contains 71,237 recordings. This project did not evaluate all 71,237 recordings. The reported results apply only to the deterministic balanced 4,000-record subset.

The subset-package SHA-256 was:

`93c7f89b74b690964c0a932502c13bf8445e38f3f16cbc0948de057b903235b6`

Evaluation was performed using:

- `backend/ml/evaluate_ast.py`
- Batch size 4.
- NVIDIA Tesla T4.
- CUDA inference.
- HuggingFace Transformers 4.44.2.
- Seed 42.

## 7. Final Unseen Evaluation Results

| Metric | Result |
|---|---:|
| Evaluated recordings | 4,000 |
| Accuracy | 92.625% |
| Precision | 98.966% |
| Recall | 86.150% |
| F1 score | 92.114% |
| Equal Error Rate | 4.775% |
| Evaluation time | 398.12 seconds |
| Throughput | 10.05 recordings/second |

### Confusion Matrix

| Actual / Predicted | Bonafide | Spoof |
|---|---:|---:|
| Bonafide | 1,982 | 18 |
| Spoof | 277 | 1,723 |

The model produced relatively few false-positive spoof detections, reflected in its 98.97% precision. However, it missed 277 spoof recordings, resulting in lower recall of 86.15%. This is the most important measured model limitation.

Evaluation evidence is stored in:

- `backend/artifacts/ast_eval_metrics.json`
- `backend/artifacts/ast_eval_confusion_matrix.png`

## 8. API and Inference Integration

The FastAPI backend exposes:

- `GET /`
- `GET /health`
- `POST /extract-features`
- `POST /predict`

The `/predict` endpoint executes the trained AST model and returns:

- Recording-level prediction.
- Confidence.
- Bonafide and spoof probabilities.
- Model version.
- Recording duration.
- Four-second segment predictions.
- Suspicious-segment indicators.
- Breathing-cadence diagnostics.

A verified API request returned model version:

`ast-asvspoof2019-la-v1`

This confirms that the deployed backend executes the trained AST checkpoint rather than the earlier baseline CNN.

## 9. Analyst Dashboard

The React and TypeScript dashboard provides:

- Drag-and-drop audio upload.
- WAV, MP3, MPEG, FLAC, and M4A support.
- Interactive waveform playback.
- Recording-level classification.
- Confidence and class probabilities.
- Active-model reporting.
- Suspicious four-second segment evidence.
- Acoustic feature summaries.
- Browser-based analysis history.
- Responsive analyst-focused layout.

The interface reports the model version returned by the backend instead of using a hard-coded model name.

## 10. Docker Deployment

Docker Compose runs two services:

### Backend

- Python 3.12 runtime.
- FastAPI served with Uvicorn.
- Port `8000`.
- Read-only model volume.
- AST model path supplied through `ACOUSTICSPACE_MODEL_PATH`.
- Health check enabled.

### Frontend

- Multi-stage Node and Nginx build.
- Port `8080`.
- Nginx forwards `/api` requests to the backend.
- Health check enabled.

Verified container state:

- Backend: healthy.
- Frontend: healthy.
- AST model loaded successfully on CPU inside the local Docker container.

The local Docker deployment uses CPU inference because GPU passthrough was not configured for the application container.

## 11. Inference Latency

Latency was measured on a short ASVspoof FLAC recording using the Dockerized CPU backend.

| Request | Time |
|---|---:|
| Cold request | 15.23 seconds |
| Warm request | 2.05 seconds |

The cold request includes model loading and initialization. The cached warm request was approximately 7.43 times faster.

These measurements apply to the tested local machine and sample. They are not general production latency guarantees.

## 12. Continuous Integration

GitHub Actions validates the repository on pushes to `main` and `week4`, and on pull requests targeting `main`.

CI jobs:

1. Backend tests
   - Installs CPU PyTorch and backend dependencies.
   - Compiles backend modules.
   - Runs Pytest.

2. Frontend checks
   - Installs Node dependencies.
   - Runs ESLint.
   - Builds the production frontend.

3. Repository safety
   - Rejects committed model binaries.
   - Validates the Docker Compose configuration.

The final CI run completed successfully for:

- Backend tests.
- Frontend checks.
- Repository safety.

The workflow uses:

- `actions/checkout@v5`
- `actions/setup-python@v6`
- `actions/setup-node@v5`

## 13. Validation Summary

Completed validation includes:

- 13 backend tests passing.
- Frontend ESLint passing.
- Frontend production build passing.
- Docker backend health check passing.
- Docker frontend health check passing.
- AST checkpoint loading locally.
- AST checkpoint loading inside Docker.
- Real `/predict` API response using AST.
- Balanced unseen evaluation smoke test.
- Balanced 4,000-record unseen GPU evaluation.
- GitHub Actions pipeline passing.
- Model binaries excluded from Git.

## 14. Known Limitations

1. The final evaluation covers a balanced 4,000-record subset rather than all 71,237 evaluation recordings.
2. Training and evaluation use ASVspoof 2019 LA, so performance may not transfer to unseen datasets, codecs, languages, microphones, rooms, or modern generation systems.
3. The final unseen recall is 86.15%, meaning some spoof recordings are classified as bonafide.
4. The AST decision is not explicitly conditioned on the independently displayed RIR, reverb, or breathing diagnostics.
5. The breathing-cadence output is a transparent signal heuristic, not a validated breathing detector.
6. The RT60 value is an acoustic proxy and may be unavailable for some recordings.
7. Recording-level probability is aggregated from four-second segment predictions.
8. The local Docker container uses CPU inference.
9. The large trained checkpoint is distributed separately and cannot be reproduced from Git alone without the dataset and training process.
10. The system is a research prototype and has not undergone forensic, adversarial, security, bias, or production-scale validation.

## 15. Final Conclusion

AcousticSpace delivers an end-to-end deepfake-audio research prototype with a real fine-tuned Audio Spectrogram Transformer, reproducible evaluation, FastAPI inference, an interactive analyst dashboard, Docker deployment, and continuous integration.

On the deterministic balanced 4,000-record subset of the official unseen ASVspoof 2019 LA evaluation split, the model achieved 92.63% accuracy, 98.97% precision, 86.15% recall, 92.11% F1, and 4.78% EER.

The results demonstrate meaningful unseen classification capability while also showing that spoof recall and cross-dataset generalization require further work. Acoustic diagnostics enhance analyst visibility but should not be represented as direct AST decision features until an explicit fusion model is implemented.