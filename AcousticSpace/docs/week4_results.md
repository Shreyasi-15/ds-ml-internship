# AcousticSpace Week 4 Results

## 1. Week 4 Scope

Week 4 focused on deploying the trained Audio Spectrogram Transformer
(AST), containerizing the application, measuring API inference latency,
refining the analyst dashboard, maintaining analysis history, and adding
continuous integration checks.

## 2. AST Training Configuration

The AST detector was fine-tuned from the Hugging Face model:

`MIT/ast-finetuned-audioset-10-10-0.4593`

Training used the ASVspoof 2019 Logical Access dataset with balanced
subsets.

| Configuration | Value |
|---|---:|
| Training samples | 5,000 |
| Training bonafide samples | 2,500 |
| Training spoof samples | 2,500 |
| Development samples | 4,000 |
| Development bonafide samples | 2,000 |
| Development spoof samples | 2,000 |
| Epochs | 3 |
| Batch size | 4 |
| Learning rate | 2e-5 |
| Sample rate | 16,000 Hz |
| Labels | `bonafide`, `spoof` |
| Model parameters | 86,190,338 |
| Training device | NVIDIA Tesla T4 |

The trained checkpoint is identified by the application as:

`ast-asvspoof2019-la-v1`

The model binary remains outside Git because it is approximately 329 MB.
The local Docker deployment mounts the checkpoint directory as a
read-only volume.

## 3. Development Metrics

The following results were measured on a balanced 4,000-sample
development subset.

| Metric | Result |
|---|---:|
| Accuracy | 99.275% |
| Precision | 98.715% |
| Recall | 99.850% |
| F1 score | 99.279% |
| Equal error rate | 0.600% |
| Evaluation loss | 0.0483 |
| Evaluation runtime | 337.38 seconds |

These are development-set measurements and must not be presented as
performance on an unseen final evaluation set.

## 4. AST Inference Integration

The FastAPI `/predict` endpoint now executes the trained AST checkpoint
instead of the Week 2 baseline CNN.

The prediction response includes:

- Predicted label
- Confidence
- Bonafide probability
- Spoof probability
- Model version
- Recording duration
- Four-second segment predictions
- Suspicious-segment indicators
- Breathing-cadence diagnostic evidence

A Docker API test returned the model version:

`ast-asvspoof2019-la-v1`

The backend performs inference on the CPU in the current local Docker
environment.

## 5. Docker Deployment

The application contains separate backend and frontend containers.

| Service | Container port | Host port |
|---|---:|---:|
| FastAPI backend | 8000 | 8000 |
| React/Nginx frontend | 80 | 8080 |

The trained AST directory is mounted into the backend container as a
read-only model volume. Docker Compose health checks confirmed that both
services were running successfully.

Verified services:

- `acousticspace-backend-1`: healthy
- `acousticspace-frontend-1`: healthy

The analyst dashboard is available at:

`http://localhost:8080`

The backend health endpoint is available at:

`http://localhost:8000/health`

## 6. Inference Latency

Latency was measured against the Dockerized `/predict` endpoint using an
ASVspoof development recording.

| Request | Latency |
|---|---:|
| Cold request | 15.23 seconds |
| Warm request | 2.05 seconds |

The cold request includes loading the approximately 329 MB AST model into
memory. Subsequent requests reuse the cached feature extractor and model.

The measured warm request was approximately 7.43 times faster than the
cold request.

Latency depends on the recording duration, CPU resources, available
memory, container state, and host system load. These measurements should
not be generalized to every input or deployment environment.

## 7. Frontend Refinement

The React analyst dashboard now provides:

- Drag-and-drop audio upload
- File validation and progress feedback
- Interactive waveform playback
- Highlighted suspicious segments
- AST prediction probabilities
- Acoustic feature reporting
- Recent analysis history stored in the browser
- Responsive dashboard layout
- Dynamic display of the backend model version

The frontend no longer presents the Week 2 CNN as the active model.

## 8. Continuous Integration

GitHub Actions validates the project on pushes to `main` and `week4` and
on pull requests targeting `main`.

The pipeline contains three jobs:

1. Backend tests
   - Installs Python dependencies
   - Compiles important backend modules
   - Runs the pytest suite

2. Frontend checks
   - Installs Node dependencies
   - Runs ESLint
   - Produces the Vite production build

3. Repository safety
   - Rejects committed ML model binaries
   - Validates the Docker Compose configuration

The Week 4 CI run completed successfully:

- Backend tests: passed
- Frontend checks: passed
- Repository safety: passed
- Total workflow duration: approximately 1 minute 20 seconds

## 9. Validation Summary

The following checks completed successfully:

- Backend pytest suite: 13 tests passed
- Frontend ESLint: passed
- Frontend production build: passed
- Docker backend health check: passed
- Docker frontend health check: passed
- AST checkpoint loading: passed
- Dockerized AST API prediction: passed
- GitHub Actions workflow: passed

## 10. Known Limitations

- The reported AST metrics are from a balanced development subset, not
  the official unseen ASVspoof evaluation split.
- AST inference currently runs on the CPU in the Docker environment.
- Cold startup is relatively slow because the 329 MB checkpoint must be
  loaded into memory.
- Latency was measured using one short development recording; a larger
  benchmark is still required.
- The trained model file is intentionally excluded from Git and must be
  installed separately before running AST inference.
- The AST classifier consumes audio spectrogram features. RIR/reverb and
  breathing-cadence values are reported as separate diagnostic evidence;
  they are not explicitly fused into the AST classification decision.
- Suspicious four-second segments are model evidence and are not a final
  forensic conclusion.
- The system is a research prototype and should not be treated as a
  production forensic tool.

## 11. Week 4 Outcome

Week 4 delivered a Dockerized AcousticSpace application with real AST
inference, a refined analyst dashboard, browser-based analysis history,
measured warm and cold latency, and automated CI validation.

The next phase is final-project validation using an unseen evaluation
subset, broader acceptance testing, and final technical documentation.