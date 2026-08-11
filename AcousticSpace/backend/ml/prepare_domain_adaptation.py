"""Validate local domain-adaptation audio and build private manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".mpeg", ".flac", ".m4a"}
LABELS = {"bonafide": 0, "spoof": 1}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_split(root: Path, split: str) -> list[dict[str, str]]:
    import librosa

    rows: list[dict[str, str]] = []
    for class_name, label in LABELS.items():
        class_dir = root / split / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing folder: {class_dir}")

        for path in sorted(class_dir.iterdir()):
            if not path.is_file() or path.name == ".gitkeep":
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported audio extension: {path}")

            try:
                duration = float(librosa.get_duration(path=path))
            except Exception as exc:
                raise ValueError(f"Audio could not be decoded: {path}") from exc

            if duration < 1.0:
                raise ValueError(f"Audio must be at least one second: {path}")

            rows.append(
                {
                    "filepath": path.relative_to(root).as_posix(),
                    "label": str(label),
                    "class_name": class_name,
                    "duration_sec": f"{duration:.3f}",
                    "sha256": sha256(path),
                }
            )
    return rows


def validate_counts(
    rows: list[dict[str, str]], split: str, minimum_per_class: int
) -> None:
    counts = {
        label: sum(int(row["label"]) == label for row in rows)
        for label in (0, 1)
    }
    if min(counts.values(), default=0) < minimum_per_class:
        raise ValueError(
            f"{split} requires at least {minimum_per_class} files per class; "
            f"found bonafide={counts[0]}, spoof={counts[1]}"
        )


def reject_leakage(
    train_rows: list[dict[str, str]], test_rows: list[dict[str, str]]
) -> None:
    train_hashes = {row["sha256"] for row in train_rows}
    overlaps = [row["filepath"] for row in test_rows if row["sha256"] in train_hashes]
    if overlaps:
        raise ValueError(
            "Identical audio appears in train and test: " + ", ".join(overlaps)
        )


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    train_rows = collect_split(root, "train")
    test_rows = collect_split(root, "test")

    validate_counts(train_rows, "train", args.min_train_per_class)
    validate_counts(test_rows, "test", args.min_test_per_class)
    reject_leakage(train_rows, test_rows)

    manifests = root / "manifests"
    write_manifest(manifests / "train.csv", train_rows)
    write_manifest(manifests / "test.csv", test_rows)

    print(f"Train recordings: {len(train_rows)}")
    print(f"Test recordings:  {len(test_rows)}")
    print(f"Manifests: {manifests}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--min-train-per-class", type=int, default=20)
    parser.add_argument("--min-test-per-class", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
