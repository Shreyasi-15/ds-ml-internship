# AcousticSpace — Week 3 Results

## Objective

Week 3 connects the trained baseline CNN to the FastAPI and React application,
adds confidence scores and suspicious segment evidence, introduces a
breathing-cadence alignment proxy, and prepares a HuggingFace Audio
Spectrogram Transformer fine-tuning workflow.

## Live baseline inference

The `/predict` endpoint loads the local `baseline_cnn.pt` checkpoint and
returns:

- Bonafide and spoof probabilities.
- Predicted label and confidence.
- Four-second segment predictions.
- Suspicious segment timestamps.
- Breathing-cadence evidence.
- Model version and classification threshold.

The checkpoint remains outside GitHub and must exist locally under
`backend/models/baseline_cnn.pt`.

## Results interface

The React dashboard now displays the baseline verdict, both class
probabilities, suspicious segment count, breathing-cadence score and model
version. Segments over the spoof threshold are highlighted in red on the
WaveSurfer waveform.

## Breathing-cadence evidence

The cadence score checks whether quiet 100–500 Hz events occur shortly after
detected speech-energy endings. This is an interpretable diagnostic proxy,
not proof of breathing and not a standalone real/fake detector.

## HuggingFace AST experiment

`backend/ml/train_ast.py` implements Week 3 fine-tuning using the pretrained
`MIT/ast-finetuned-audioset-10-10-0.4593` Audio Spectrogram Transformer.

### AST smoke-run result

A controlled AST fine-tuning smoke run completed successfully.

- Training samples: 64
- Development samples: 32
- Epochs: 1
- Batch size: 2
- Training loss: 0.0308
- Development loss: 0.00000118
- Training runtime: approximately 22 minutes 36 seconds
- Saved locally at: `backend/models/ast_week3/best`
- Execution log: `backend/artifacts/ast_smoke_run.txt`

The saved transformer is approximately 345 MB and remains outside GitHub.

This smoke run validates the fine-tuning workflow. It is not a full-dataset
transformer benchmark, and its development loss should not be presented as
final model accuracy.

## Verification checklist

- [x] Local CNN checkpoint is non-empty.
- [x] `/health` reports Week 3.
- [x] `/predict` returns probabilities and segments.
- [x] Suspicious segments appear on the waveform.
- [x] Backend tests pass: 13 passed.
- [x] Frontend lint passes.
- [x] Frontend production build passes.
- [x] AST smoke fine-tuning completes.
