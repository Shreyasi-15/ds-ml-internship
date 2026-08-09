"""Shared data utilities for balanced Wav2Vec2 anti-spoofing experiments."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


SAMPLE_RATE = 16_000
CLIP_SECONDS = 4.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)


def read_manifest(path: Path) -> list[dict[str, str]]:
    """Read and minimally validate an AcousticSpace CSV manifest."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    required = {"filepath", "label"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Invalid or empty manifest: {path}")

    for row in rows:
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError(f"Unexpected label {label} in {path}")

    return rows


def balanced_rows(
    rows: list[dict[str, str]],
    maximum: int | None,
    seed: int,
) -> list[dict[str, str]]:
    """Select the same number of bonafide and spoof examples."""
    grouped = {0: [], 1: []}
    for row in rows:
        grouped[int(row["label"])].append(row)

    available_per_class = min(len(grouped[0]), len(grouped[1]))
    if maximum is None:
        samples_per_class = available_per_class
    else:
        if maximum < 2 or maximum % 2:
            raise ValueError("maximum must be an even integer of at least 2")
        samples_per_class = maximum // 2

    if samples_per_class > available_per_class:
        raise ValueError(
            f"Requested {samples_per_class} per class, but only "
            f"{available_per_class} are available"
        )

    generator = random.Random(seed)
    selected: list[dict[str, str]] = []
    for label in (0, 1):
        candidates = grouped[label].copy()
        generator.shuffle(candidates)
        selected.extend(candidates[:samples_per_class])

    generator.shuffle(selected)
    return selected


def _fit_clip(
    waveform: np.ndarray,
    training: bool,
    generator: random.Random,
) -> np.ndarray:
    """Crop or zero-pad audio to the model's fixed four-second input."""
    if len(waveform) > CLIP_SAMPLES:
        maximum_start = len(waveform) - CLIP_SAMPLES
        start = generator.randint(0, maximum_start) if training else maximum_start // 2
        waveform = waveform[start : start + CLIP_SAMPLES]
    elif len(waveform) < CLIP_SAMPLES:
        waveform = np.pad(waveform, (0, CLIP_SAMPLES - len(waveform)))

    return waveform.astype(np.float32, copy=False)


def _augment(
    waveform: np.ndarray,
    generator: random.Random,
) -> np.ndarray:
    """Apply lightweight gain and noise augmentation without extra packages."""
    waveform = waveform * generator.uniform(0.75, 1.25)

    if generator.random() < 0.35:
        signal_rms = float(np.sqrt(np.mean(waveform**2) + 1e-12))
        snr_db = generator.uniform(15.0, 30.0)
        noise_rms = signal_rms / (10 ** (snr_db / 20.0))
        numpy_generator = np.random.default_rng(generator.randrange(2**32))
        waveform = waveform + numpy_generator.normal(
            0.0, noise_rms, waveform.shape
        ).astype(np.float32)

    return np.clip(waveform, -1.0, 1.0).astype(np.float32)


class Wav2Vec2ManifestDataset(Dataset):
    """Load balanced manifest rows as fixed-length raw waveforms."""

    def __init__(
        self,
        manifest_path: Path,
        dataset_root: Path,
        feature_extractor,
        maximum: int | None,
        seed: int,
        training: bool,
    ) -> None:
        self.dataset_root = dataset_root.resolve()
        self.feature_extractor = feature_extractor
        self.rows = balanced_rows(read_manifest(manifest_path), maximum, seed)
        self.seed = seed
        self.training = training

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        audio_path = (self.dataset_root / row["filepath"]).resolve()
        audio_path.relative_to(self.dataset_root)

        waveform, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        if waveform.size == 0:
            raise ValueError(f"Empty audio: {audio_path}")

        generator = random.Random(self.seed + index)
        waveform = _fit_clip(waveform, self.training, generator)
        if self.training:
            waveform = _augment(waveform, generator)

        inputs = self.feature_extractor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            max_length=CLIP_SAMPLES,
            truncation=True,
            return_attention_mask=True,
        )

        return {
            "input_values": inputs["input_values"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
        }
