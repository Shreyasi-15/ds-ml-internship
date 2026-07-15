"""PyTorch dataset loader for ASVspoof 2019 LA recordings."""

import csv
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


SAMPLE_RATE = 16_000
DURATION_SECONDS = 4.0
NUMBER_OF_MELS = 64
FFT_SIZE = 1024
HOP_LENGTH = 512


def audio_to_log_mel(
    audio_path: str | Path,
    sample_rate: int = SAMPLE_RATE,
    duration_seconds: float = DURATION_SECONDS,
    number_of_mels: int = NUMBER_OF_MELS,
) -> torch.Tensor:
    """Load one recording and convert it to a log-mel spectrogram."""

    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file was not found: {audio_path}")

    # Librosa converts stereo recordings to mono and resamples them.
    waveform, _ = librosa.load(
        audio_path,
        sr=sample_rate,
        mono=True,
    )

    required_samples = int(sample_rate * duration_seconds)

    # Short recordings are padded; long recordings are cut.
    waveform = librosa.util.fix_length(
        waveform,
        size=required_samples,
    )

    mel_spectrogram = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=FFT_SIZE,
        hop_length=HOP_LENGTH,
        n_mels=number_of_mels,
        power=2.0,
    )

    log_mel = librosa.power_to_db(
        mel_spectrogram,
        ref=np.max,
    )

    # Normalize each spectrogram for more stable CNN training.
    mean = float(log_mel.mean())
    standard_deviation = float(log_mel.std())

    if standard_deviation > 1e-6:
        log_mel = (log_mel - mean) / standard_deviation
    else:
        log_mel = log_mel - mean

    # Add a channel dimension: [1, mel bands, time frames].
    return torch.from_numpy(log_mel).float().unsqueeze(0)


class ASVspoofDataset(Dataset):
    """Read an ASVspoof CSV manifest and return tensors and labels."""

    def __init__(
        self,
        manifest_path: str | Path,
        dataset_root: str | Path,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.dataset_root = Path(dataset_root).resolve()

        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest was not found: {self.manifest_path}"
            )

        if not self.dataset_root.is_dir():
            raise NotADirectoryError(
                f"Dataset root was not found: {self.dataset_root}"
            )

        with self.manifest_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as manifest_file:
            reader = csv.DictReader(manifest_file)

            required_columns = {"filepath", "label"}
            available_columns = set(reader.fieldnames or [])

            missing_columns = required_columns - available_columns

            if missing_columns:
                raise ValueError(
                    "Manifest is missing columns: "
                    + ", ".join(sorted(missing_columns))
                )

            self.rows = list(reader)

        if not self.rows:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]

        audio_path = (
            self.dataset_root / row["filepath"]
        ).resolve()

        # Prevent an invalid manifest path from escaping dataset_root.
        try:
            audio_path.relative_to(self.dataset_root)
        except ValueError as error:
            raise ValueError(
                f"Audio path is outside the dataset: {audio_path}"
            ) from error

        spectrogram = audio_to_log_mel(audio_path)

        label_value = int(row["label"])

        if label_value not in {0, 1}:
            raise ValueError(
                f"Invalid label {label_value} at row {index}"
            )

        label = torch.tensor(label_value, dtype=torch.long)

        return spectrogram, label