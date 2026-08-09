"""Fine-tune a balanced Wav2Vec2 audio anti-spoofing classifier."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoFeatureExtractor,
    Trainer,
    TrainingArguments,
    Wav2Vec2ForSequenceClassification,
)

from wav2vec2_data import Wav2Vec2ManifestDataset


MODEL_NAME = "facebook/wav2vec2-base"
BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent


def metrics(prediction) -> dict[str, float]:
    logits = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
    labels = np.asarray(prediction.label_ids, dtype=int)
    predicted = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def main(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        label2id={"bonafide": 0, "spoof": 1},
        id2label={0: "bonafide", 1: "spoof"},
        ignore_mismatched_sizes=True,
    )
    model.freeze_feature_encoder()

    train_dataset = Wav2Vec2ManifestDataset(
        args.train_manifest, args.dataset_root, extractor,
        args.max_train_samples, seed=42, training=True,
    )
    dev_dataset = Wav2Vec2ManifestDataset(
        args.dev_manifest, args.dataset_root, extractor,
        args.max_dev_samples, seed=43, training=False,
    )

    print(f"Balanced training samples: {len(train_dataset):,}")
    print(f"Balanced development samples: {len(dev_dataset):,}")

    training_options = dict(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        warmup_ratio=0.1,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_total_limit=1,
        fp16=args.fp16,
        dataloader_num_workers=args.num_workers,
        logging_steps=20,
        report_to=[],
        seed=42,
    )
    strategy_parameter = (
        "eval_strategy"
        if "eval_strategy"
        in inspect.signature(TrainingArguments.__init__).parameters
        else "evaluation_strategy"
    )
    training_options[strategy_parameter] = "epoch"
    training_args = TrainingArguments(**training_options)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=metrics,
    )
    trainer.train()

    best_dir = args.output_dir / "best"
    trainer.save_model(str(best_dir))
    extractor.save_pretrained(str(best_dir))
    evaluation = trainer.evaluate()
    trainer.save_metrics("eval", evaluation)

    with (args.output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "base_model": MODEL_NAME,
                "train_samples": len(train_dataset),
                "dev_samples": len(dev_dataset),
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "metrics": evaluation,
            },
            handle,
            indent=2,
        )

    print(f"Saved Wav2Vec2 checkpoint: {best_dir}")
    print(evaluation)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--train-manifest", type=Path,
        default=PROJECT_ROOT / "dataset" / "manifests" / "train.csv",
    )
    parser.add_argument(
        "--dev-manifest", type=Path,
        default=PROJECT_ROOT / "dataset" / "manifests" / "dev.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=BACKEND_ROOT / "models" / "wav2vec2_training",
    )
    parser.add_argument("--max-train-samples", type=int, default=2000)
    parser.add_argument("--max-dev-samples", type=int, default=500)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
