"""Evaluation metrics for the AcousticSpace baseline model."""

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


def calculate_eer(
    labels: Sequence[int],
    spoof_probabilities: Sequence[float],
) -> tuple[float, float]:
    """
    Calculate Equal Error Rate and its decision threshold.

    Label 0 represents bonafide.
    Label 1 represents spoof.
    """

    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(spoof_probabilities, dtype=np.float64)

    if labels_array.size == 0:
        raise ValueError("Labels cannot be empty.")

    if labels_array.size != scores_array.size:
        raise ValueError(
            "Labels and spoof probabilities must have the same length."
        )

    if set(np.unique(labels_array)) != {0, 1}:
        raise ValueError(
            "EER requires both bonafide and spoof recordings."
        )

    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels_array,
        scores_array,
        pos_label=1,
    )

    false_negative_rate = 1.0 - true_positive_rate

    index = int(
        np.nanargmin(
            np.abs(false_positive_rate - false_negative_rate)
        )
    )

    eer = (
        false_positive_rate[index] + false_negative_rate[index]
    ) / 2.0

    threshold = thresholds[index]

    return float(eer), float(threshold)


def calculate_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
    spoof_probabilities: Sequence[float],
) -> dict:
    """Calculate classification metrics for the spoof class."""

    labels_array = np.asarray(labels, dtype=np.int64)
    predictions_array = np.asarray(predictions, dtype=np.int64)
    scores_array = np.asarray(spoof_probabilities, dtype=np.float64)

    if labels_array.size == 0:
        raise ValueError("Metrics cannot be calculated without labels.")

    if not (
        labels_array.size
        == predictions_array.size
        == scores_array.size
    ):
        raise ValueError(
            "Labels, predictions and probabilities "
            "must have the same length."
        )

    if not set(np.unique(labels_array)).issubset({0, 1}):
        raise ValueError("Labels must contain only 0 and 1.")

    if not set(np.unique(predictions_array)).issubset({0, 1}):
        raise ValueError("Predictions must contain only 0 and 1.")

    eer, eer_threshold = calculate_eer(
        labels_array,
        scores_array,
    )

    matrix = confusion_matrix(
        labels_array,
        predictions_array,
        labels=[0, 1],
    )

    return {
        "accuracy": float(
            accuracy_score(labels_array, predictions_array)
        ),
        "precision": float(
            precision_score(
                labels_array,
                predictions_array,
                pos_label=1,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labels_array,
                predictions_array,
                pos_label=1,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                labels_array,
                predictions_array,
                pos_label=1,
                zero_division=0,
            )
        ),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "confusion_matrix": matrix.tolist(),
    }