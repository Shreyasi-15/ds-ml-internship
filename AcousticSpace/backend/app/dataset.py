"""ASVspoof-style dataset curation and manifest generation."""

import argparse
import csv
import json
from pathlib import Path


VALID_LABELS = {"bonafide": 0, "spoof": 1}


class DatasetProtocolError(ValueError):
    """Raised when an ASVspoof-style protocol row is invalid."""


def build_manifest(
    protocol_path: str,
    audio_dir: str,
    output_csv: str,
    audio_ext: str = ".flac",
    split: str = "unspecified",
    summary_json: str | None = None,
    source_name: str = "ASVspoof-style",
    strict_files: bool = True,
):
    """Build a validated manifest and optional dataset summary."""
    protocol = Path(protocol_path)
    audio_root = Path(audio_dir)
    output = Path(output_csv)
    rows: list[dict[str, str | int]] = []
    missing_files: list[str] = []

    with protocol.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 5:
                raise DatasetProtocolError(
                    f"Line {line_number} has {len(parts)} fields; expected at least 5"
                )

            speaker_id = parts[0]
            filename = parts[1]
            attack_id = parts[-2]
            label_name = parts[-1].lower()
            if label_name not in VALID_LABELS:
                raise DatasetProtocolError(
                    f"Line {line_number} has unsupported label {label_name!r}"
                )

            audio_filename = filename if Path(filename).suffix else filename + audio_ext
            filepath = audio_root / audio_filename
            if not filepath.is_file():
                missing_files.append(str(filepath))

            rows.append(
                {
                    "filepath": filepath.as_posix(),
                    "label": VALID_LABELS[label_name],
                    "label_name": label_name,
                    "speaker_id": speaker_id,
                    "attack_id": attack_id,
                    "split": split,
                }
            )

    if strict_files and missing_files:
        preview = ", ".join(missing_files[:3])
        raise FileNotFoundError(
            f"{len(missing_files)} referenced audio file(s) are missing: {preview}"
        )
    if not rows:
        raise DatasetProtocolError("Protocol did not contain any usable rows")

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["filepath", "label", "label_name", "speaker_id", "attack_id", "split"]
    with output.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "source_dataset": source_name,
        "split": split,
        "total": len(rows),
        "bonafide": sum(row["label"] == 0 for row in rows),
        "spoof": sum(row["label"] == 1 for row in rows),
        "missing_files": len(missing_files),
        "manifest": output.as_posix(),
    }
    if summary_json:
        summary_path = Path(summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(
        f"Manifest written to {output}: {summary['bonafide']} bonafide, "
        f"{summary['spoof']} spoof"
    )
    return output.as_posix()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build an ASVspoof-style manifest")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--output", default="manifest.csv")
    parser.add_argument("--audio-ext", default=".flac")
    parser.add_argument("--split", default="unspecified")
    parser.add_argument("--summary")
    parser.add_argument("--source-name", default="ASVspoof-style")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    build_manifest(
        args.protocol,
        args.audio_dir,
        args.output,
        audio_ext=args.audio_ext,
        split=args.split,
        summary_json=args.summary,
        source_name=args.source_name,
        strict_files=not args.allow_missing,
    )
