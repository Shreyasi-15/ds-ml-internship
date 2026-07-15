"""Generate CSV manifests from the ASVspoof 2019 LA protocols."""

import argparse
import csv
from collections import Counter
from pathlib import Path


SPLITS = {
    "train": {
        "audio_directory": "ASVspoof2019_LA_train/flac",
        "protocol": (
            "ASVspoof2019_LA_cm_protocols/"
            "ASVspoof2019.LA.cm.train.trn.txt"
        ),
    },
    "dev": {
        "audio_directory": "ASVspoof2019_LA_dev/flac",
        "protocol": (
            "ASVspoof2019_LA_cm_protocols/"
            "ASVspoof2019.LA.cm.dev.trl.txt"
        ),
    },
    "eval": {
        "audio_directory": "ASVspoof2019_LA_eval/flac",
        "protocol": (
            "ASVspoof2019_LA_cm_protocols/"
            "ASVspoof2019.LA.cm.eval.trl.txt"
        ),
    },
}


def build_manifest(
    dataset_root: Path,
    output_directory: Path,
    split: str,
) -> None:
    """Create one CSV manifest from an official protocol file."""

    configuration = SPLITS[split]
    protocol_path = dataset_root / configuration["protocol"]
    audio_directory = Path(configuration["audio_directory"])

    if not protocol_path.is_file():
        raise FileNotFoundError(
            f"Protocol file was not found: {protocol_path}"
        )

    rows = []
    missing_files = []

    with protocol_path.open("r", encoding="utf-8") as protocol_file:
        for line_number, line in enumerate(protocol_file, start=1):
            values = line.strip().split()

            if not values:
                continue

            if len(values) != 5:
                raise ValueError(
                    f"Invalid protocol row at {protocol_path}:"
                    f"{line_number}"
                )

            speaker_id, recording_id, _, attack_id, label_name = values

            if label_name == "bonafide":
                label = 0
            elif label_name == "spoof":
                label = 1
            else:
                raise ValueError(
                    f"Unknown label '{label_name}' on line {line_number}"
                )

            relative_audio_path = (
                audio_directory / f"{recording_id}.flac"
            )
            absolute_audio_path = dataset_root / relative_audio_path

            if not absolute_audio_path.is_file():
                missing_files.append(absolute_audio_path)

            rows.append(
                {
                    "filepath": relative_audio_path.as_posix(),
                    "label": label,
                    "label_name": label_name,
                    "speaker_id": speaker_id,
                    "attack_id": attack_id,
                    "split": split,
                }
            )

    if missing_files:
        examples = "\n".join(
            str(path) for path in missing_files[:5]
        )
        raise FileNotFoundError(
            f"{len(missing_files)} audio files are missing.\n{examples}"
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{split}.csv"

    fieldnames = [
        "filepath",
        "label",
        "label_name",
        "speaker_id",
        "attack_id",
        "split",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["label_name"] for row in rows)

    print(f"Created: {output_path}")
    print(f"Total: {len(rows):,}")
    print(f"Bonafide: {counts['bonafide']:,}")
    print(f"Spoof: {counts['spoof']:,}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ASVspoof 2019 LA manifests."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to the extracted ASVspoof2019 LA directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/manifests"),
        help="Directory where CSV manifests will be created.",
    )
    arguments = parser.parse_args()

    dataset_root = arguments.dataset_root.expanduser().resolve()
    output_directory = arguments.output_dir.expanduser().resolve()

    for split in SPLITS:
        build_manifest(dataset_root, output_directory, split)


if __name__ == "__main__":
    main()