# AcousticSpace Final Project Report

## 1. Project Overview

AcousticSpace is an end-to-end research prototype for bonafide-versus-spoof audio classification. The final application uses a domain-adapted Wav2Vec2 sequence classifier, a FastAPI backend, a React and TypeScript analyst dashboard, Librosa acoustic analysis, authenticated SQLite history, Docker Compose deployment, and GitHub Actions validation.

The detector provides decision-support evidence and must not be treated as a guaranteed forensic verdict.

## 2. Objectives

The project objectives were to:

1. prepare reproducible ASVspoof training and evaluation data
2. train a transformer classifier for bonafide/spoof detection
3. identify and reduce real-world domain shift
4. integrate the trained model into an API and analyst dashboard
5. display transparent recording, segment, waveform, and acoustic evidence
6. retain authenticated analysis history safely
7. validate the implementation and keep private audio and model binaries outside Git

## 3. Final Architecture

Uploaded audio follows this workflow:

1. The React frontend validates the selected file.
2. FastAPI authenticates the user and validates the upload.
3. Librosa decodes mono audio at 16 kHz.
4. The Wav2Vec2 processor prepares waveform input.
5. `Wav2Vec2ForSequenceClassification` returns two-class logits.
6. Softmax produces bonafide and spoof probabilities.
7. Four-second windows produce suspicious-segment evidence.
8. Librosa calculates independent acoustic measurements.
9. The dashboard displays the result, waveform, metadata, features, statistics, and history.
10. SQLite stores the completed analysis under the authenticated user.

The final model identifier is `wav2vec2-asvspoof-domain-v2`.

## 4. Dataset

The base dataset is ASVspoof 2019 Logical Access. Balanced CSV manifests reference audio paths and binary labels:

- `0`: bonafide
- `1`: spoof

To address domain shift, the project added private real-world bonafide and AI-generated recordings:

- domain training: 20 recordings per class
- held-out domain test: 5 recordings per class

The preparation script validates minimum class counts and rejects byte-identical audio shared between training and testing. Private recordings are ignored by Git.

## 5. Wav2Vec2 Training

Training used a Colab Tesla T4 environment with PyTorch, HuggingFace Transformers, Scikit-learn, and Librosa. The mixed bundle added balanced domain-training recordings to the balanced ASVspoof bundle while keeping the domain test isolated.

The training run completed three epochs and saved the best checkpoint. The checkpoint is distributed separately because the model binary is approximately 346 MB when packaged with results.

Required checkpoint directory:

```text
AcousticSpace/backend/models/wav2vec2_asvspoof_domain_v2/best
```

## 6. Evaluation

### 6.1 Balanced ASVspoof evaluation

The held-out evaluation contained 1,000 recordings, balanced across bonafide and spoof classes.

| Metric | Result |
|---|---:|
| Accuracy | 99.0% |
| Precision | 99.395% |
| Recall | 98.6% |
| F1 score | 98.996% |
| Equal Error Rate | 0.9% |

Confusion matrix:

| Actual / Predicted | Bonafide | Spoof |
|---|---:|---:|
| Bonafide | 497 | 3 |
| Spoof | 7 | 493 |

### 6.2 Private domain test

The held-out private domain test contained five recordings per class.

| Metric | Result |
|---|---:|
| Accuracy | 100% |
| Precision | 100% |
| Recall | 100% |
| F1 score | 100% |
| Equal Error Rate | 0% |

Confusion matrix: `[[5, 0], [0, 5]]`

The 100% domain result must be interpreted cautiously. Ten recordings are insufficient for a reliable generalization claim.

## 7. Inference Integration

The FastAPI `/predict` endpoint loads the separately installed Wav2Vec2 checkpoint and returns:

- predicted label
- confidence
- bonafide and spoof probabilities
- model version
- duration
- four-second segment predictions
- suspicious-segment flags
- breathing-cadence evidence

Verified local smoke tests classified a held-out genuine recording as bonafide and a held-out AI-generated recording as spoof.

## 8. Acoustic Evidence

The `/extract-features` endpoint and dashboard report:

- RT60 proxy
- reverb ratio
- breathing-band ratio
- duration
- peak amplitude
- RMS energy
- spectral centroid
- spectral bandwidth
- mel-spectrogram shape

These features change with each recording. They are calculated independently and are not explicitly fused into the Wav2Vec2 classifier.

## 9. Dashboard

The analyst dashboard includes:

- authenticated access
- drag-and-drop audio upload
- waveform playback
- Wav2Vec2 prediction and probability bars
- probability doughnut chart
- suspicious-segment highlighting
- audio metadata and acoustic evidence
- aggregate model statistics
- confidence distribution and seven-day prediction chart
- recent server history with Keep and Delete actions
- persistent saved history

## 10. History and Retention

SQLite stores analysis metadata per authenticated user. New unkept records are automatically removed after 30 days. A user can keep a record indefinitely or delete any recent or saved record. Queries and mutations always include the authenticated user identifier.

## 11. Deployment

Docker Compose builds the FastAPI backend and React/Nginx frontend. The local model directory is mounted read-only, and the backend receives:

```text
ACOUSTICSPACE_MODEL_PATH=/app/models/wav2vec2_asvspoof_domain_v2/best
```

The checkpoint must exist locally before starting the backend container.

## 12. Continuous Integration

GitHub Actions performs:

1. Python dependency installation, active-module compilation, and backend tests
2. frontend dependency installation, ESLint, and production build
3. rejection of committed model binaries and Docker Compose configuration validation

## 13. Validation Summary

- 14 backend tests passed
- frontend ESLint passed
- frontend production build passed
- Python source compilation passed
- Git whitespace validation passed
- local genuine and spoof Wav2Vec2 inference passed
- private recordings and checkpoints remained outside Git

## 14. Limitations

1. The domain test contains only 10 recordings.
2. Performance may change across datasets, modern generators, languages, codecs, devices, rooms, and noise conditions.
3. Acoustic evidence is displayed independently and is not fused into Wav2Vec2.
4. The breathing and RT60 values are transparent signal proxies rather than validated forensic estimators.
5. Four-second segment predictions are supporting evidence.
6. CPU inference can be slow, especially for long recordings.
7. The large checkpoint is excluded from Git and must be distributed separately.
8. The system has not undergone production security, adversarial, bias, or forensic validation.

## 15. Conclusion

AcousticSpace delivers a functioning domain-adapted Wav2Vec2 deepfake-audio research prototype with real inference, transparent acoustic evidence, an interactive analyst dashboard, authenticated retained history, reproducible preparation and evaluation utilities, Docker deployment, and continuous integration.

The balanced 1,000-record ASVspoof evaluation achieved 99.0% accuracy and 0.9% EER. The 10-record private domain test achieved perfect classification, but its small size remains the most important limitation. Broader real-world evaluation is required before any forensic or production claim.
