"""Optional Week 3 HuggingFace Audio Spectrogram Transformer fine-tuning.

Start with a small smoke run. Do not report AST accuracy until an evaluation
run has completed and its metrics have been saved.
"""

import argparse
import csv
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
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            self.rows = list(csv.DictReader(handle))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        audio_path = (self.dataset_root / row["filepath"]).resolve()
        audio_path.relative_to(self.dataset_root)
        waveform, _ = librosa.load(audio_path, sr=16_000, mono=True)
        inputs = self.feature_extractor(
            waveform,
            sampling_rate=16_000,
            return_tensors="pt",
        )
        return {
            "input_values": inputs["input_values"].squeeze(0),
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
        }


def deterministic_subset(dataset: Dataset, maximum: int | None) -> Dataset:
    if maximum is None or maximum >= len(dataset):
        return dataset
    return Subset(dataset, list(range(maximum)))


def main(arguments: argparse.Namespace) -> None:
    feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_NAME)
    model = ASTForAudioClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        label2id={"bonafide": 0, "spoof": 1},
        id2label={0: "bonafide", 1: "spoof"},
        ignore_mismatched_sizes=True,
    )

    train_dataset = deterministic_subset(
        ASTManifestDataset(
            arguments.train_manifest,
            arguments.dataset_root,
            feature_extractor,
        ),
        arguments.max_train_samples,
    )
    dev_dataset = deterministic_subset(
        ASTManifestDataset(
            arguments.dev_manifest,
            arguments.dataset_root,
            feature_extractor,
        ),
        arguments.max_dev_samples,
    )

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
    trainer.save_model(str(arguments.output_dir / "best"))
    feature_extractor.save_pretrained(str(arguments.output_dir / "best"))
    metrics = trainer.evaluate()
    print(metrics)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "manifests" / "train.csv",
    )
    parser.add_argument(
        "--dev-manifest",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "manifests" / "dev.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BACKEND_ROOT / "models" / "ast_week3",
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-dev-samples", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_arguments())
