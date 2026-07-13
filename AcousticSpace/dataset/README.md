# Dataset curation plan

## Selected training target

AcousticSpace uses the **ASVspoof Logical Access (LA)** family as the initial
bonafide-versus-spoof source. The original recordings are not redistributed in
this repository. Obtain them from the official ASVspoof release under its
license and keep them inside `dataset/raw/` or `dataset/ASVspoof*/`, which are
ignored by Git.

## Manifest schema

| Field | Meaning |
|---|---|
| `filepath` | Audio path used by the training loader |
| `label` | `0` for bonafide, `1` for spoof |
| `label_name` | Human-readable label |
| `speaker_id` | Speaker/group identifier used to prevent leakage |
| `attack_id` | Synthesis/conversion system identifier when available |
| `split` | Train, development, evaluation, or demo |

Create a manifest after placing a protocol and FLAC files locally:

```bash
python backend/app/dataset.py ^
  --protocol dataset/raw/ASVspoof_protocol.txt ^
  --audio-dir dataset/raw/flac ^
  --output backend/data/manifests/train.csv ^
  --summary backend/data/manifests/train_summary.json ^
  --split train ^
  --source-name "ASVspoof LA"
```

The command fails when labels are invalid or referenced files are missing. Do
not train until the summary reports `missing_files: 0`.

## Included demo data

`dataset/demo` contains four generated WAV files and a protocol used only to
prove that dataset indexing and the audio pipeline work. It is not real speech,
must not be used for accuracy, and does not replace ASVspoof training data.

Regenerate it from the project root:

```bash
python scripts/generate_demo_dataset.py
python backend/app/dataset.py --protocol dataset/demo/protocol.txt --audio-dir dataset/demo/audio --audio-ext .wav --output dataset/demo/manifest.csv --summary dataset/demo/dataset_summary.json --split demo --source-name "Synthetic smoke test"
```

## Leakage rules for Week 2

- Never place the same source recording in multiple splits.
- Group related samples by speaker and source file.
- Keep evaluation speakers, rooms/codecs, and attack systems unseen when
  possible.
- Report class counts and missing files for every manifest.
