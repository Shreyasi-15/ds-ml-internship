"""Tests for the Week 2 baseline CNN."""

import torch

from ml.model import BaselineCNN


def test_baseline_cnn_output_shape():
    model = BaselineCNN()
    model.eval()

    spectrograms = torch.randn(4, 1, 64, 126)

    with torch.inference_mode():
        logits = model(spectrograms)

    assert logits.shape == (4, 2)


def test_baseline_cnn_probabilities_are_valid():
    model = BaselineCNN()
    model.eval()

    spectrograms = torch.randn(3, 1, 64, 126)

    with torch.inference_mode():
        probabilities = model.predict_probabilities(spectrograms)

    assert probabilities.shape == (3, 2)
    assert torch.all(probabilities >= 0)
    assert torch.all(probabilities <= 1)
    assert torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(3),
        atol=1e-6,
    )

    predicted_labels = probabilities.argmax(dim=1)
    assert set(predicted_labels.tolist()).issubset({0, 1})