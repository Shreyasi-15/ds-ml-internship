"""Evaluate the trained AcousticSpace baseline CNN."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader, Dataset, Subset

from ml.dataset import ASVspoofDataset
from ml.metrics import calculate_metrics
from ml.model import BaselineCNN


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent

DEFAULT_MANIFEST = (
    PROJECT_ROOT / "dataset" / "manifests" / "eval.csv"
)
DEFAULT_MODEL_PATH = (
    BACKEND_ROOT / "models" / "baseline_cnn.pt"
)
DEFAULT_METRICS_PATH = (
    BACKEND_ROOT / "artifacts" / "metrics.json"
)
DEFAULT_CONFUSION_MATRIX_PATH = (
    BACKEND_ROOT / "artifacts" / "confusion_matrix.png"
)


def limit_dataset(
    dataset: Dataset,
    maximum_samples: int | None,
    seed: int,
) -> Dataset:
    """Create a deterministic random subset for smoke testing."""

    if maximum_samples is None or maximum_samples >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(
        len(dataset),
        generator=generator,
    )[:maximum_samples].tolist()

    return Subset(dataset, indices)


def save_confusion_matrix(
    matrix: list[list[int]],
    output_path: Path,
) -> None:
    """Save the confusion matrix as a PNG image."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6, 5))

    display = ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(matrix),
        display_labels=["Bonafide", "Spoof"],
    )
    display.plot(
        ax=axis,
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )

    axis.set_title("AcousticSpace Baseline CNN")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def evaluate(arguments: argparse.Namespace) -> None:
    """Evaluate a saved model checkpoint."""

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    if not arguments.model_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint was not found: {arguments.model_path}"
        )

    full_dataset = ASVspoofDataset(
        arguments.manifest,
        arguments.dataset_root,
    )
    evaluation_dataset = limit_dataset(
        full_dataset,
        arguments.max_samples,
        arguments.seed,
    )

    print(f"Evaluation recordings used: {len(evaluation_dataset):,}")

    data_loader = DataLoader(
        evaluation_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=0,
    )

    checkpoint = torch.load(
        arguments.model_path,
        map_location=device,
        weights_only=True,
    )

    model = BaselineCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    labels: list[int] = []
    predictions: list[int] = []
    spoof_probabilities: list[float] = []

    with torch.inference_mode():
        for batch_number, (spectrograms, batch_labels) in enumerate(
            data_loader,
            start=1,
        ):
            spectrograms = spectrograms.to(device)

            logits = model(spectrograms)
            probabilities = torch.softmax(logits, dim=1)
            batch_predictions = logits.argmax(dim=1)

            labels.extend(batch_labels.tolist())
            predictions.extend(batch_predictions.cpu().tolist())
            spoof_probabilities.extend(
                probabilities[:, 1].cpu().tolist()
            )

            if batch_number % 100 == 0 or batch_number == len(data_loader):
                print(
                    f"Processed batch {batch_number}/{len(data_loader)}"
                )

    results = calculate_metrics(
        labels,
        predictions,
        spoof_probabilities,
    )

    results["evaluated_recordings"] = len(labels)
    results["checkpoint_epoch"] = checkpoint.get("epoch")
    results["checkpoint_dev_loss"] = checkpoint.get("dev_loss")
    results["checkpoint_dev_accuracy"] = checkpoint.get(
        "dev_accuracy"
    )
    results["model_version"] = checkpoint.get(
        "model_version",
        "baseline-cnn-v1",
    )

    arguments.metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with arguments.metrics_path.open(
        "w",
        encoding="utf-8",
    ) as metrics_file:
        json.dump(results, metrics_file, indent=2)

    save_confusion_matrix(
        results["confusion_matrix"],
        arguments.confusion_matrix_path,
    )

    print()
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 score:  {results['f1_score']:.4f}")
    print(f"EER:       {results['eer']:.4f}")
    print(f"Metrics:   {arguments.metrics_path}")
    print(
        "Confusion matrix: "
        f"{arguments.confusion_matrix_path}"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the AcousticSpace baseline CNN."
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
        default=DEFAULT_CONFUSION_MATRIX_PATH,
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_arguments())