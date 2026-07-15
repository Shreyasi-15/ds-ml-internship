"""Baseline CNN for ASVspoof log-mel spectrogram classification."""

import torch
from torch import nn


class BaselineCNN(nn.Module):
    """Small CNN that classifies audio as bonafide or spoof."""

    def __init__(self, number_of_classes: int = 2) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Input: [batch, 1, 64, 126]
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Converts any remaining spectrogram size to 1 × 1.
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.30),
            nn.Linear(64, number_of_classes),
        )

    def forward(self, spectrograms: torch.Tensor) -> torch.Tensor:
        """Return unnormalized class scores called logits."""

        if spectrograms.ndim != 4:
            raise ValueError(
                "Expected input shape "
                "[batch, channel, mel bands, time frames]."
            )

        if spectrograms.shape[1] != 1:
            raise ValueError(
                "Expected one spectrogram input channel."
            )

        features = self.features(spectrograms)
        return self.classifier(features)

    def predict_probabilities(
        self,
        spectrograms: torch.Tensor,
    ) -> torch.Tensor:
        """Return bonafide and spoof probabilities."""

        logits = self.forward(spectrograms)
        return torch.softmax(logits, dim=1)