"""Evaluate a trained AcousticSpace AST checkpoint.

This script performs inference only. It does not update model weights.
It supports deterministic balanced subsets for reproducible evaluation.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from transformers import (
    ASTFeatureExtractor,
    ASTForAudioClassification,
)

from ml.metrics import calculate_metrics
from ml.train_ast import (
    ASTManifestDataset,
    balanced_subset,
)


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent

DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "dataset"
    / "manifests"
    / "eval.csv"
)

DEFAULT_MODEL_PATH = (
    BACKEND_ROOT
    / "models"
    / "ast_asvspoof_v1"
    / "best"
)

DEFAULT_METRICS_PATH = (
    BACKEND_ROOT
    / "artifacts"
    / "ast_eval_metrics.json"
)

DEFAULT_CONFUSION_MATRIX_PATH = (
    BACKEND_ROOT
    / "artifacts"
    / "ast_eval_confusion_matrix.png"
)

MODEL_VERSION = "ast-asvspoof2019-la-v1"


def save_confusion_matrix(
    matrix: list[list[int]],
    output_path: Path,
) -> None:
    """Save a labeled AST confusion matrix."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(6, 5),
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(matrix),
        display_labels=[
            "Bonafide",
            "Spoof",
        ],
    )

    display.plot(
        ax=axis,
        cmap="Purples",
        colorbar=False,
        values_format="d",
    )

    axis.set_title(
        "AcousticSpace AST Evaluation"
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
    )

    plt.close(figure)


def evaluate(
    arguments: argparse.Namespace,
) -> None:
    """Evaluate the AST checkpoint without training."""

    if not arguments.manifest.is_file():
        raise FileNotFoundError(
            "Evaluation manifest was not found: "
            f"{arguments.manifest}"
        )

    if not arguments.dataset_root.is_dir():
        raise FileNotFoundError(
            "Dataset root was not found: "
            f"{arguments.dataset_root}"
        )

    required_model_files = [
        arguments.model_path / "config.json",
        arguments.model_path / "model.safetensors",
        (
            arguments.model_path
            / "preprocessor_config.json"
        ),
    ]

    missing_model_files = [
        path
        for path in required_model_files
        if not path.is_file()
    ]

    if missing_model_files:
        missing_text = "\n".join(
            str(path)
            for path in missing_model_files
        )

        raise FileNotFoundError(
            "The AST checkpoint is incomplete:\n"
            f"{missing_text}"
        )

    if arguments.max_samples < 2:
        raise ValueError(
            "At least two evaluation samples are required."
        )

    if arguments.max_samples % 2 != 0:
        raise ValueError(
            "--max-samples must be an even number "
            "for balanced evaluation."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Checkpoint: {arguments.model_path}")
    print(f"Manifest: {arguments.manifest}")
    print(
        "Maximum balanced samples: "
        f"{arguments.max_samples:,}"
    )

    feature_extractor = (
        ASTFeatureExtractor.from_pretrained(
            arguments.model_path
        )
    )

    model = (
        ASTForAudioClassification.from_pretrained(
            arguments.model_path
        )
    )

    model = model.to(device)
    model.eval()

    complete_dataset = ASTManifestDataset(
        arguments.manifest,
        arguments.dataset_root,
        feature_extractor,
    )

    evaluation_dataset = balanced_subset(
        complete_dataset,
        arguments.max_samples,
        seed=arguments.seed,
    )

    data_loader = DataLoader(
        evaluation_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.num_workers,
        pin_memory=device.type == "cuda",
    )

    labels: list[int] = []
    predictions: list[int] = []
    spoof_probabilities: list[float] = []

    started_at = time.perf_counter()

    with torch.inference_mode():
        for batch_number, batch in enumerate(
            data_loader,
            start=1,
        ):
            input_values = batch[
                "input_values"
            ].to(device)

            batch_labels = batch[
                "labels"
            ]

            outputs = model(
                input_values=input_values
            )

            logits = outputs.logits

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            batch_predictions = torch.argmax(
                logits,
                dim=1,
            )

            labels.extend(
                batch_labels.cpu().tolist()
            )

            predictions.extend(
                batch_predictions.cpu().tolist()
            )

            spoof_probabilities.extend(
                probabilities[
                    :,
                    1,
                ].cpu().tolist()
            )

            if (
                batch_number % 25 == 0
                or batch_number
                == len(data_loader)
            ):
                print(
                    "Processed batch "
                    f"{batch_number}/"
                    f"{len(data_loader)}"
                )

    elapsed_seconds = (
        time.perf_counter()
        - started_at
    )

    results = calculate_metrics(
        labels,
        predictions,
        spoof_probabilities,
    )

    label_counts = Counter(labels)

    results.update(
        {
            "model_version": MODEL_VERSION,
            "checkpoint_path": str(
                arguments.model_path
            ),
            "manifest_path": str(
                arguments.manifest
            ),
            "dataset_split": "eval",
            "subset_type": (
                "deterministic_balanced"
            ),
            "seed": arguments.seed,
            "evaluated_recordings": len(
                labels
            ),
            "bonafide_recordings": (
                label_counts[0]
            ),
            "spoof_recordings": (
                label_counts[1]
            ),
            "batch_size": (
                arguments.batch_size
            ),
            "device": str(device),
            "evaluation_seconds": round(
                elapsed_seconds,
                4,
            ),
            "samples_per_second": round(
                len(labels)
                / elapsed_seconds,
                4,
            ),
        }
    )

    arguments.metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with arguments.metrics_path.open(
        "w",
        encoding="utf-8",
    ) as metrics_file:
        json.dump(
            results,
            metrics_file,
            indent=2,
        )

    save_confusion_matrix(
        results["confusion_matrix"],
        arguments.confusion_matrix_path,
    )

    print()
    print("AST evaluation completed.")
    print(
        "Recordings: "
        f"{results['evaluated_recordings']:,}"
    )
    print(
        "Bonafide:  "
        f"{results['bonafide_recordings']:,}"
    )
    print(
        "Spoof:     "
        f"{results['spoof_recordings']:,}"
    )
    print(
        "Accuracy:  "
        f"{results['accuracy']:.4f}"
    )
    print(
        "Precision: "
        f"{results['precision']:.4f}"
    )
    print(
        "Recall:    "
        f"{results['recall']:.4f}"
    )
    print(
        "F1 score:  "
        f"{results['f1_score']:.4f}"
    )
    print(
        "EER:       "
        f"{results['eer']:.4f}"
    )
    print(
        "Time:      "
        f"{elapsed_seconds:.2f} seconds"
    )
    print(
        "Metrics:   "
        f"{arguments.metrics_path}"
    )
    print(
        "Confusion matrix: "
        f"{arguments.confusion_matrix_path}"
    )


def parse_arguments() -> argparse.Namespace:
    """Parse AST evaluation arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the trained AcousticSpace "
            "AST checkpoint."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )

    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=DEFAULT_METRICS_PATH,
    )

    parser.add_argument(
        "--confusion-matrix-path",
        type=Path,
        default=(
            DEFAULT_CONFUSION_MATRIX_PATH
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=4000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_arguments())