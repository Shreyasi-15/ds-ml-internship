"""Build a compact balanced ASVspoof bundle for Colab training."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from wav2vec2_data import balanced_rows, read_manifest


SPLITS = (
    ("train", 42),
    ("dev", 43),
    ("eval", 44),
)


def copy_split(
    name: str,
    seed: int,
    manifest: Path,
    maximum: int,
    dataset_root: Path,
    output_dir: Path,
) -> int:
    """Write one subset manifest and copy its referenced recordings."""
    rows = balanced_rows(read_manifest(manifest), maximum, seed)
    if not rows:
        raise ValueError(f"No usable rows found in {manifest}")

    bundle_dataset = output_dir / "dataset"
    bundle_manifests = output_dir / "manifests"
    bundle_manifests.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(rows, start=1):
        relative_path = Path(row["filepath"])
        source = (dataset_root / relative_path).resolve()
        source.relative_to(dataset_root)
        if not source.is_file():
            raise FileNotFoundError(f"Missing audio: {source}")

        destination = bundle_dataset / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

        if index % 250 == 0 or index == len(rows):
            print(f"{name}: copied {index:,}/{len(rows):,}")

    output_manifest = bundle_manifests / f"{name}.csv"
    with output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main(args: argparse.Namespace) -> None:
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists: {output_dir}. Choose a new path or remove it intentionally."
        )

    sizes = {
        "train": args.train_samples,
        "dev": args.dev_samples,
        "eval": args.eval_samples,
    }
    manifests = {
        "train": args.train_manifest,
        "dev": args.dev_manifest,
        "eval": args.eval_manifest,
    }

    output_dir.mkdir(parents=True)
    counts: dict[str, int] = {}
    try:
        for name, seed in SPLITS:
            counts[name] = copy_split(
                name=name,
                seed=seed,
                manifest=manifests[name],
                maximum=sizes[name],
                dataset_root=dataset_root,
                output_dir=output_dir,
            )

        archive = shutil.make_archive(
            str(output_dir),
            "zip",
            root_dir=output_dir.parent,
            base_dir=output_dir.name,
        )
    except Exception:
        print(
            "Bundle creation stopped. The partial output was preserved for inspection: "
            f"{output_dir}"
        )
        raise

    print()
    print("Balanced Wav2Vec2 bundle completed.")
    print(f"Train: {counts['train']:,}")
    print(f"Dev:   {counts['dev']:,}")
    print(f"Eval:  {counts['eval']:,}")
    print(f"Folder: {output_dir}")
    print(f"ZIP:    {archive}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--dev-samples", type=int, default=500)
    parser.add_argument("--eval-samples", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
