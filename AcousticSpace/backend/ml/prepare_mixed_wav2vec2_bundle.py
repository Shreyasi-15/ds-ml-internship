"""Build a Colab bundle mixing ASVspoof data with private domain recordings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"filepath", "label"}.issubset(rows[0]):
        raise ValueError(f"Invalid or empty manifest: {path}")
    return rows


def audio_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_balanced(rows: list[dict[str, str]], name: str) -> None:
    counts = {label: sum(int(row["label"]) == label for row in rows) for label in (0, 1)}
    if counts[0] != counts[1]:
        raise ValueError(f"{name} is not balanced: bonafide={counts[0]}, spoof={counts[1]}")


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filepath", "label"])
        writer.writeheader()
        writer.writerows(
            {"filepath": row["filepath"], "label": row["label"]} for row in rows
        )


def copy_domain_rows(
    rows: list[dict[str, str]], domain_root: Path, output_dataset: Path
) -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    for row in rows:
        relative = Path(row["filepath"])
        source = (domain_root / relative).resolve()
        source.relative_to(domain_root)
        if not source.is_file():
            raise FileNotFoundError(f"Missing domain audio: {source}")

        bundled_relative = Path("domain_adaptation") / relative
        destination = output_dataset / bundled_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append({"filepath": bundled_relative.as_posix(), "label": row["label"]})
    return copied


def resolve_audio(dataset_root: Path, row: dict[str, str]) -> Path:
    path = (dataset_root / row["filepath"]).resolve()
    path.relative_to(dataset_root)
    if not path.is_file():
        raise FileNotFoundError(f"Missing bundled audio: {path}")
    return path


def reject_test_leakage(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    dataset_root: Path,
) -> None:
    train_hashes = {audio_hash(resolve_audio(dataset_root, row)) for row in train_rows}
    overlaps = [
        row["filepath"]
        for row in test_rows
        if audio_hash(resolve_audio(dataset_root, row)) in train_hashes
    ]
    if overlaps:
        raise ValueError(
            "Audio leakage between mixed training and domain test: " + ", ".join(overlaps)
        )


def main(args: argparse.Namespace) -> None:
    base = args.base_bundle.resolve()
    domain = args.domain_root.resolve()
    output = args.output_dir.resolve()

    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}. Remove it intentionally or choose another path."
        )
    if not (base / "dataset").is_dir() or not (base / "manifests").is_dir():
        raise FileNotFoundError(f"Invalid base Wav2Vec2 bundle: {base}")
    if not (domain / "manifests" / "train.csv").is_file():
        raise FileNotFoundError("Run prepare_domain_adaptation.py first")

    output.mkdir(parents=True)
    try:
        shutil.copytree(base / "dataset", output / "dataset")
        output_manifests = output / "manifests"

        base_train = read_rows(base / "manifests" / "train.csv")
        base_dev = read_rows(base / "manifests" / "dev.csv")
        base_eval = read_rows(base / "manifests" / "eval.csv")
        domain_train = read_rows(domain / "manifests" / "train.csv")
        domain_test = read_rows(domain / "manifests" / "test.csv")

        copied_train = copy_domain_rows(domain_train, domain, output / "dataset")
        copied_test = copy_domain_rows(domain_test, domain, output / "dataset")

        mixed_train = base_train + copied_train
        validate_balanced(mixed_train, "mixed train")
        validate_balanced(base_dev, "development")
        validate_balanced(base_eval, "ASVspoof evaluation")
        validate_balanced(copied_test, "domain test")
        reject_test_leakage(mixed_train, copied_test, output / "dataset")

        write_rows(output_manifests / "train.csv", mixed_train)
        write_rows(output_manifests / "dev.csv", base_dev)
        write_rows(output_manifests / "eval.csv", base_eval)
        write_rows(output_manifests / "domain_test.csv", copied_test)

        archive = shutil.make_archive(
            str(output), "zip", root_dir=output.parent, base_dir=output.name
        )
    except Exception:
        print(f"Build failed; partial output retained for inspection: {output}")
        raise

    print("Mixed Wav2Vec2 bundle completed.")
    print(f"Train:       {len(mixed_train):,}")
    print(f"Dev:         {len(base_dev):,}")
    print(f"ASV eval:    {len(base_eval):,}")
    print(f"Domain test: {len(copied_test):,}")
    print(f"Folder: {output}")
    print(f"ZIP:    {archive}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-bundle", type=Path, required=True)
    parser.add_argument("--domain-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
