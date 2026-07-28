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

`backend/ml/train_ast.py` provides the Week 3 fine-tuning workflow using
`MIT/ast-finetuned-audioset-10-10-0.4593`.

Record the following only after running the experiment:

- Training and development sample counts.
- Number of epochs.
- Final development loss.
- Evaluation accuracy, precision, recall, F1 and EER.
- Hardware and runtime.

Do not claim that the transformer was fully fine-tuned until the training
command completes and its result artifacts are saved.

## Verification checklist

- [ ] Local CNN checkpoint is non-empty.
- [ ] `/health` reports Week 3.
- [ ] `/predict` returns probabilities and segments.
- [ ] Suspicious segments appear on the waveform.
- [ ] Backend tests pass.
- [ ] Frontend lint passes.
- [ ] Frontend production build passes.
- [ ] AST smoke fine-tuning completes.
