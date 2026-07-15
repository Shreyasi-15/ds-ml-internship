"""Train the AcousticSpace baseline CNN."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Subset

from ml.dataset import ASVspoofDataset
from ml.model import BaselineCNN


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent

DEFAULT_TRAIN_MANIFEST = (
    PROJECT_ROOT / "dataset" / "manifests" / "train.csv"
)
DEFAULT_DEV_MANIFEST = (
    PROJECT_ROOT / "dataset" / "manifests" / "dev.csv"
)
DEFAULT_MODEL_PATH = (
    BACKEND_ROOT / "models" / "baseline_cnn.pt"
)
DEFAULT_HISTORY_PATH = (
    BACKEND_ROOT / "artifacts" / "training_history.json"
)


def set_random_seed(seed: int) -> None:
    """Make training results more reproducible."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limit_dataset(
    dataset: Dataset,
    maximum_samples: int | None,
    seed: int,
) -> Dataset:
    """Create a random subset for fast smoke testing."""

    if maximum_samples is None or maximum_samples >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(
        len(dataset),
        generator=generator,
    )[:maximum_samples].tolist()

    return Subset(dataset, indices)


def calculate_class_weights(
    dataset: ASVspoofDataset,
) -> torch.Tensor:
    """Compensate for the dataset containing more spoof recordings."""

    counts = torch.zeros(2, dtype=torch.long)

    for row in dataset.rows:
        counts[int(row["label"])] += 1

    if torch.any(counts == 0):
        raise ValueError(
            "The training manifest must contain both classes."
        )

    weights = counts.sum().float() / (2.0 * counts.float())

    print(
        "Training classes — "
        f"bonafide: {counts[0].item():,}, "
        f"spoof: {counts[1].item():,}"
    )
    print(
        "Class weights — "
        f"bonafide: {weights[0].item():.4f}, "
        f"spoof: {weights[1].item():.4f}"
    )

    return weights


def run_epoch(
    model: BaselineCNN,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: Adam | None = None,
) -> tuple[float, float]:
    """Run one training or validation epoch."""

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0

    for spectrograms, labels in data_loader:
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(spectrograms)
            loss = loss_function(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        correct_predictions += (
            logits.argmax(dim=1) == labels
        ).sum().item()
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("The data loader contains no recordings.")

    average_loss = total_loss / total_examples
    accuracy = correct_predictions / total_examples

    return average_loss, accuracy


def save_history(history: dict, output_path: Path) -> None:
    """Save training statistics as JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as history_file:
        json.dump(history, history_file, indent=2)


def train(arguments: argparse.Namespace) -> None:
    """Train the CNN and save the best checkpoint."""

    set_random_seed(arguments.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    full_train_dataset = ASVspoofDataset(
        arguments.train_manifest,
        arguments.dataset_root,
    )
    full_dev_dataset = ASVspoofDataset(
        arguments.dev_manifest,
        arguments.dataset_root,
    )

    train_dataset = limit_dataset(
        full_train_dataset,
        arguments.max_train_samples,
        arguments.seed,
    )
    dev_dataset = limit_dataset(
        full_dev_dataset,
        arguments.max_dev_samples,
        arguments.seed + 1,
    )

    print(f"Training recordings used: {len(train_dataset):,}")
    print(f"Development recordings used: {len(dev_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=arguments.batch_size,
        shuffle=True,
        num_workers=0,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = BaselineCNN().to(device)

    class_weights = calculate_class_weights(
        full_train_dataset
    ).to(device)

    loss_function = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = Adam(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )

    history = {
        "configuration": {
            "dataset_root": str(arguments.dataset_root),
            "train_manifest": str(arguments.train_manifest),
            "dev_manifest": str(arguments.dev_manifest),
            "batch_size": arguments.batch_size,
            "learning_rate": arguments.learning_rate,
            "weight_decay": arguments.weight_decay,
            "maximum_epochs": arguments.epochs,
            "patience": arguments.patience,
            "seed": arguments.seed,
            "device": str(device),
        },
        "epochs": [],
    }

    arguments.model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_dev_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, arguments.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model=model,
            data_loader=train_loader,
            loss_function=loss_function,
            device=device,
            optimizer=optimizer,
        )

        dev_loss, dev_accuracy = run_epoch(
            model=model,
            data_loader=dev_loader,
            loss_function=loss_function,
            device=device,
        )

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "dev_loss": dev_loss,
            "dev_accuracy": dev_accuracy,
        }
        history["epochs"].append(epoch_result)

        print(
            f"Epoch {epoch}/{arguments.epochs} — "
            f"train loss: {train_loss:.4f}, "
            f"train accuracy: {train_accuracy:.4f}, "
            f"dev loss: {dev_loss:.4f}, "
            f"dev accuracy: {dev_accuracy:.4f}"
        )

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "dev_loss": dev_loss,
                    "dev_accuracy": dev_accuracy,
                    "class_names": ["bonafide", "spoof"],
                    "model_version": "baseline-cnn-v1",
                },
                arguments.model_path,
            )

            print(f"Saved best model: {arguments.model_path}")
        else:
            epochs_without_improvement += 1

        save_history(history, arguments.history_path)

        if epochs_without_improvement >= arguments.patience:
            print(
                "Early stopping: validation loss stopped improving."
            )
            break

    print(f"Best validation loss: {best_dev_loss:.4f}")
    print(f"Training history: {arguments.history_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the AcousticSpace baseline CNN."
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=DEFAULT_TRAIN_MANIFEST,
    )
    parser.add_argument(
        "--dev-manifest",
        type=Path,
        default=DEFAULT_DEV_MANIFEST,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=DEFAULT_HISTORY_PATH,
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-dev-samples", type=int)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_arguments())