"""Create a reproducible balanced ASVspoof evaluation package."""

import argparse
import csv
import hashlib
import random
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=4000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    arguments = parser.parse_args()

    if arguments.samples < 2:
        raise ValueError(
            "At least two samples are required."
        )

    if arguments.samples % 2 != 0:
        raise ValueError(
            "Sample count must be even."
        )

    dataset_root = (
        arguments.dataset_root
        .expanduser()
        .resolve()
    )

    manifest = (
        arguments.manifest
        .expanduser()
        .resolve()
    )

    output_directory = (
        arguments.output_directory
        .expanduser()
        .resolve()
    )

    with manifest.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    rows_by_label = {
        0: [],
        1: [],
    }

    for row in rows:
        label = int(row["label"])

        if label not in rows_by_label:
            raise ValueError(
                f"Unexpected label: {label}"
            )

        rows_by_label[label].append(row)

    samples_per_class = (
        arguments.samples // 2
    )

    if any(
        len(label_rows) < samples_per_class
        for label_rows
        in rows_by_label.values()
    ):
        raise ValueError(
            "Not enough recordings in both classes."
        )

    generator = random.Random(
        arguments.seed
    )

    selected_rows = []

    for label in (0, 1):
        label_rows = (
            rows_by_label[label].copy()
        )

        generator.shuffle(label_rows)

        selected_rows.extend(
            label_rows[:samples_per_class]
        )

    generator.shuffle(selected_rows)

    if output_directory.exists():
        shutil.rmtree(output_directory)

    packaged_dataset_root = (
        output_directory / "LA"
    )

    packaged_dataset_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for number, row in enumerate(
        selected_rows,
        start=1,
    ):
        relative_path = Path(
            row["filepath"]
        )

        source = (
            dataset_root / relative_path
        )

        destination = (
            packaged_dataset_root
            / relative_path
        )

        if not source.is_file():
            raise FileNotFoundError(
                f"Missing audio file: {source}"
            )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            source,
            destination,
        )

        if (
            number % 500 == 0
            or number == len(selected_rows)
        ):
            print(
                f"Copied {number:,}/"
                f"{len(selected_rows):,}"
            )

    subset_manifest = (
        output_directory
        / "eval_subset.csv"
    )

    fieldnames = list(
        selected_rows[0].keys()
    )

    with subset_manifest.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(selected_rows)

    archive_base = output_directory.parent / (
        output_directory.name
    )

    archive_path = Path(
        shutil.make_archive(
            str(archive_base),
            "zip",
            root_dir=output_directory.parent,
            base_dir=output_directory.name,
        )
    )

    digest = hashlib.sha256()

    with archive_path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    print()
    print("Package completed.")
    print(
        "Bonafide recordings:",
        samples_per_class,
    )
    print(
        "Spoof recordings:",
        samples_per_class,
    )
    print("Manifest:", subset_manifest)
    print("Archive:", archive_path)
    print(
        "Archive size:",
        round(
            archive_path.stat().st_size
            / (1024**2),
            2,
        ),
        "MB",
    )
    print("SHA-256:", digest.hexdigest())


if __name__ == "__main__":
    main()