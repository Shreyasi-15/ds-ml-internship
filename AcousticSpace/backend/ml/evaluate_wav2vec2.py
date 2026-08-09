"""Evaluate a trained Wav2Vec2 checkpoint on a balanced held-out subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_curve
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

from train_wav2vec2 import metrics
from wav2vec2_data import Wav2Vec2ManifestDataset


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = AutoFeatureExtractor.from_pretrained(args.model_path)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(args.model_path)
    model.to(device).eval()

    dataset = Wav2Vec2ManifestDataset(
        args.manifest, args.dataset_root, extractor,
        args.max_samples, seed=args.seed, training=False,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    labels: list[int] = []
    logits_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            batch_labels = batch.pop("labels")
            outputs = model(**{key: value.to(device) for key, value in batch.items()})
            labels.extend(batch_labels.tolist())
            logits_batches.append(outputs.logits.cpu().numpy())

    logits = np.concatenate(logits_batches)
    label_array = np.asarray(labels)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    predicted = np.argmax(logits, axis=1)

    class Prediction:
        predictions = logits
        label_ids = label_array

    results = metrics(Prediction())
    results["confusion_matrix"] = confusion_matrix(
        label_array, predicted, labels=[0, 1]
    ).tolist()

    fpr, tpr, thresholds = roc_curve(label_array, probabilities[:, 1], pos_label=1)
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fnr - fpr)))
    results["eer"] = float((fpr[index] + fnr[index]) / 2.0)
    results["eer_threshold"] = float(thresholds[index])
    results["evaluated_recordings"] = len(dataset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(json.dumps(results, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path,
        default=PROJECT_ROOT / "dataset" / "manifests" / "eval.csv",
    )
    parser.add_argument(
        "--model-path", type=Path,
        default=BACKEND_ROOT / "models" / "wav2vec2_training" / "best",
    )
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument(
        "--output", type=Path,
        default=BACKEND_ROOT / "artifacts" / "wav2vec2_eval_metrics.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
