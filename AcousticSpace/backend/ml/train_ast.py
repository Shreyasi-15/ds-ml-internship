"""Balanced HuggingFace Audio Spectrogram Transformer fine-tuning.

Start with a small balanced smoke run. Do not report AST accuracy until a
proper evaluation run has completed and its metrics have been saved.
"""

import argparse
import csv
import random
from pathlib import Path

import librosa
import torch
from torch.utils.data import Dataset, Subset
from transformers import (
    ASTFeatureExtractor,
    ASTForAudioClassification,
    Trainer,
    TrainingArguments,
)


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"


class ASTManifestDataset(Dataset):
    """Read an AcousticSpace manifest and create AST inputs."""

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

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]

        audio_path = (
            self.dataset_root / row["filepath"]
        ).resolve()

        # Prevent manifest paths from escaping the dataset directory.
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
            "input_values": inputs["input_values"].squeeze(0),
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
    """Select equal bonafide and spoof samples reproducibly."""

    if maximum is None or maximum >= len(dataset):
        return dataset

    if maximum < 2:
        raise ValueError(
            "A balanced subset requires at least two samples."
        )

    if maximum % 2 != 0:
        raise ValueError(
            "The maximum sample count must be an even number."
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

    if any(
        len(indices) < samples_per_class
        for indices in indices_by_label.values()
    ):
        raise ValueError(
            "The dataset does not contain enough samples "
            "from both classes."
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


def main(arguments: argparse.Namespace) -> None:
    arguments.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Loading pretrained AST model: {MODEL_NAME}")

    feature_extractor = ASTFeatureExtractor.from_pretrained(
        MODEL_NAME
    )

    model = ASTForAudioClassification.from_pretrained(
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

    print(f"Training samples used: {len(train_dataset)}")
    print(f"Development samples used: {len(dev_dataset)}")

    training_arguments = TrainingArguments(
        output_dir=str(arguments.output_dir),
        learning_rate=arguments.learning_rate,
        per_device_train_batch_size=arguments.batch_size,
        per_device_eval_batch_size=arguments.batch_size,
        num_train_epochs=arguments.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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
    )

    trainer.train()

    best_path = arguments.output_dir / "best"

    trainer.save_model(str(best_path))
    feature_extractor.save_pretrained(str(best_path))

    metrics = trainer.evaluate()

    print("Balanced AST smoke run completed.")
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
        default=BACKEND_ROOT / "models" / "ast_week3",
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