# AcousticSpace — Week 2 Results

## Objective

Week 2 focused on preparing the ASVspoof dataset, training and evaluating a baseline CNN, and adding an interactive audio waveform to the frontend.

## Dataset

The baseline uses the ASVspoof 2019 Logical Access dataset. The complete dataset is stored outside the repository at:

`C:\Users\shrey\Datasets\ASVspoof2019\LA`

The generated manifests contain:

| Split | Recordings |
|---|---:|
| Train | 25,380 |
| Development | 24,844 |
| Evaluation | 71,237 |

The full audio dataset and ZIP archive are excluded from GitHub.

## Preprocessing

Each recording is:

- Loaded as mono audio.
- Resampled to 16 kHz.
- Cut or padded to four seconds.
- Converted into a 64-band log-mel spectrogram.
- Normalized before being returned to PyTorch.

Labels are represented as:

- `0` — bonafide
- `1` — spoof

## Baseline model

The baseline CNN contains three convolution blocks with batch normalization, ReLU activation and max pooling. Adaptive average pooling and dropout are applied before the final two-class output layer.

Class weighting is used to account for the imbalance between bonafide and spoof recordings. Training includes validation monitoring and early stopping.

The best saved checkpoint reached approximately 95.64% development accuracy during the completed training epochs.

## Evaluation results

The saved model was evaluated on all 71,237 official evaluation recordings.

| Metric | Result |
|---|---:|
| Accuracy | 87.65% |
| Precision | 99.10% |
| Recall | 87.01% |
| F1 score | 92.66% |
| Equal Error Rate | 10.41% |

Generated artifacts:

- `backend/artifacts/metrics.json`
- `backend/artifacts/confusion_matrix.png`
- `backend/artifacts/training_history.json`

The model checkpoint is stored locally at `backend/models/baseline_cnn.pt` and is excluded from GitHub.

## Frontend visualization

WaveSurfer.js was integrated into the React frontend. It provides:

- Interactive waveform rendering.
- Play and pause controls.
- Current playback time and duration.
- Automatic cleanup when another file is selected.

A 15.14 MB, 90-second WAV recording was successfully uploaded, rendered and analyzed during the mid-review architecture check.

## Verification

- Backend test suite: 12 passed.
- Frontend lint: passed.
- Frontend production build: passed.
- Large-file FastAPI response: HTTP 200.
- Full evaluation completed on 71,237 recordings.

See `mid_review_validation.md` for controlled acoustic validation and large-file evidence.

## Evidence boundary

The RT60, temporal-decay and breathing-band values are diagnostic acoustic proxies. They should not be described as direct measurements of a true physical room impulse response or as standalone proof that an audio recording is real or fake.

The baseline CNN provides the initial classification benchmark. Advanced transformer and prediction-interface development remain planned for Week 3.

