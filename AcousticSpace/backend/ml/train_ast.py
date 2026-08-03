"""Balanced HuggingFace AST fine-tuning and evaluation.

Start with a balanced smoke run. Do not report AST performance until an
evaluation run has completed and its metrics have been saved.
"""

import argparse
import csv
import random
from pathlib import Path

import librosa
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
)
from torch.utils.data import Dataset, Subset
from transformers import (
    ASTFeatureExtractor,
    ASTForAudioClassification,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent

MODEL_NAME = (
    "MIT/ast-finetuned-audioset-10-10-0.4593"
)


class ASTManifestDataset(Dataset):
    """Read a manifest and create AST model inputs."""

    def __init__(
        self,
        manifest_path: Path,
        dataset_root: Path,
        feature_extractor: ASTFeatureExtractor,
    ) -> None:
        self.dataset_root = dataset_root.resolve()
        self.feature_extractor = feature_extractor

        with manifest_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            self.rows = list(csv.DictReader(handle))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:
        row = self.rows[index]

        audio_path = (
            self.dataset_root / row["filepath"]
        ).resolve()

        # Prevent manifest paths from escaping the dataset root.
        audio_path.relative_to(self.dataset_root)

        waveform, _ = librosa.load(
            audio_path,
            sr=16_000,
            mono=True,
        )

        inputs = self.feature_extractor(
            waveform,
            sampling_rate=16_000,
            return_tensors="pt",
        )

        return {
            "input_values": (
                inputs["input_values"].squeeze(0)
            ),
            "labels": torch.tensor(
                int(row["label"]),
                dtype=torch.long,
            ),
        }


def balanced_subset(
    dataset: ASTManifestDataset,
    maximum: int | None,
    seed: int = 42,
) -> Dataset:
    """Select equal numbers from both classes reproducibly."""

    if maximum is None:
        return dataset

    if maximum < 2:
        raise ValueError(
            "A balanced subset requires at least two samples."
        )

    if maximum % 2 != 0:
        raise ValueError(
            "The maximum sample count must be even."
        )

    if maximum > len(dataset):
        raise ValueError(
            "The requested subset is larger than the dataset."
        )

    indices_by_label: dict[int, list[int]] = {
        0: [],
        1: [],
    }

    for index, row in enumerate(dataset.rows):
        label = int(row["label"])

        if label not in indices_by_label:
            raise ValueError(
                f"Unexpected label {label} in the manifest."
            )

        indices_by_label[label].append(index)

    samples_per_class = maximum // 2

    for label, indices in indices_by_label.items():
        if len(indices) < samples_per_class:
            raise ValueError(
                f"Label {label} has only {len(indices)} samples; "
                f"{samples_per_class} are required."
            )

    random_generator = random.Random(seed)
    selected_indices: list[int] = []

    for label in (0, 1):
        label_indices = indices_by_label[label].copy()
        random_generator.shuffle(label_indices)

        selected_indices.extend(
            label_indices[:samples_per_class]
        )

    random_generator.shuffle(selected_indices)

    print(
        f"Balanced subset: {samples_per_class} bonafide, "
        f"{samples_per_class} spoof"
    )

    return Subset(dataset, selected_indices)


def compute_classification_metrics(
    prediction: EvalPrediction,
) -> dict[str, float]:
    """Calculate two-class AST evaluation metrics."""

    logits = prediction.predictions

    if isinstance(logits, tuple):
        logits = logits[0]

    labels = np.asarray(
        prediction.label_ids,
        dtype=int,
    )

    predicted_labels = np.argmax(
        logits,
        axis=1,
    )

    shifted_logits = (
        logits
        - np.max(
            logits,
            axis=1,
            keepdims=True,
        )
    )

    exponentials = np.exp(shifted_logits)

    probabilities = (
        exponentials
        / np.sum(
            exponentials,
            axis=1,
            keepdims=True,
        )
    )

    spoof_probabilities = probabilities[:, 1]

    accuracy = accuracy_score(
        labels,
        predicted_labels,
    )

    precision, recall, f1_score, _ = (
        precision_recall_fscore_support(
            labels,
            predicted_labels,
            average="binary",
            zero_division=0,
        )
    )

    false_positive_rate, true_positive_rate, _ = (
        roc_curve(
            labels,
            spoof_probabilities,
            pos_label=1,
        )
    )

    false_negative_rate = 1 - true_positive_rate

    equal_error_index = int(
        np.nanargmin(
            np.abs(
                false_negative_rate
                - false_positive_rate
            )
        )
    )

    equal_error_rate = float(
        (
            false_positive_rate[equal_error_index]
            + false_negative_rate[equal_error_index]
        )
        / 2
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1_score),
        "eer": equal_error_rate,
    }


def main(arguments: argparse.Namespace) -> None:
    arguments.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Loading pretrained AST model: {MODEL_NAME}"
    )

    feature_extractor = (
        ASTFeatureExtractor.from_pretrained(
            MODEL_NAME
        )
    )

    model = (
        ASTForAudioClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            label2id={
                "bonafide": 0,
                "spoof": 1,
            },
            id2label={
                0: "bonafide",
                1: "spoof",
            },
            ignore_mismatched_sizes=True,
        )
    )

    complete_train_dataset = ASTManifestDataset(
        arguments.train_manifest,
        arguments.dataset_root,
        feature_extractor,
    )

    complete_dev_dataset = ASTManifestDataset(
        arguments.dev_manifest,
        arguments.dataset_root,
        feature_extractor,
    )

    train_dataset = balanced_subset(
        complete_train_dataset,
        arguments.max_train_samples,
        seed=42,
    )

    dev_dataset = balanced_subset(
        complete_dev_dataset,
        arguments.max_dev_samples,
        seed=43,
    )

    print(
        f"Training samples used: {len(train_dataset)}"
    )

    print(
        f"Development samples used: {len(dev_dataset)}"
    )

    training_arguments = TrainingArguments(
        output_dir=str(arguments.output_dir),
        learning_rate=arguments.learning_rate,
        per_device_train_batch_size=(
            arguments.batch_size
        ),
        per_device_eval_batch_size=(
            arguments.batch_size
        ),
        num_train_epochs=arguments.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=10,
        report_to=[],
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=(
            compute_classification_metrics
        ),
    )

    trainer.train()

    best_path = arguments.output_dir / "best"

    trainer.save_model(str(best_path))

    feature_extractor.save_pretrained(
        str(best_path)
    )

    metrics = trainer.evaluate()

    trainer.save_metrics(
        "eval",
        metrics,
    )

    trainer.save_state()

    print("Balanced AST training completed.")
    print(f"Saved model: {best_path}")
    print(f"Evaluation metrics: {metrics}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=(
            PROJECT_ROOT
            / "dataset"
            / "manifests"
            / "train.csv"
        ),
    )

    parser.add_argument(
        "--dev-manifest",
        type=Path,
        default=(
            PROJECT_ROOT
            / "dataset"
            / "manifests"
            / "dev.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            BACKEND_ROOT
            / "models"
            / "ast_training"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
    )

    parser.add_argument(
        "--max-train-samples",
        type=int,
    )

    parser.add_argument(
        "--max-dev-samples",
        type=int,
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_arguments())