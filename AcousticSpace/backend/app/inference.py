"""Wav2Vec2 inference for AcousticSpace deepfake-audio detection."""

from functools import lru_cache
import os
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2ForSequenceClassification,
)


BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = (
    BACKEND_ROOT
    / "models"
    / "wav2vec2_asvspoof_domain_v2"
    / "best"
)

SAMPLE_RATE = 16_000
SEGMENT_SECONDS = 4.0
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_SECONDS)
N_FFT = 1024
SPOOF_THRESHOLD = 0.50
DEFAULT_BATCH_SIZE = 4
MODEL_VERSION = "wav2vec2-asvspoof-domain-v2"


class ModelUnavailableError(RuntimeError):
    """Raised when the trained Wav2Vec2 checkpoint cannot be used."""


def _checkpoint_path() -> Path:
    """Return the configured Wav2Vec2 checkpoint directory."""

    configured = os.getenv("ACOUSTICSPACE_MODEL_PATH")

    if configured:
        return Path(configured).expanduser()

    return DEFAULT_MODEL_PATH


def _inference_batch_size() -> int:
    """Read and validate the Wav2Vec2 inference batch size."""

    configured = os.getenv(
        "ACOUSTICSPACE_INFERENCE_BATCH_SIZE",
        str(DEFAULT_BATCH_SIZE),
    )

    try:
        batch_size = int(configured)
    except ValueError as exc:
        raise ModelUnavailableError(
            "ACOUSTICSPACE_INFERENCE_BATCH_SIZE must be an integer"
        ) from exc

    if batch_size < 1:
        raise ModelUnavailableError(
            "ACOUSTICSPACE_INFERENCE_BATCH_SIZE must be at least 1"
        )

    return batch_size


@lru_cache(maxsize=1)
def load_prediction_model() -> tuple[
    AutoFeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    torch.device,
]:
    """Load and cache the trained Wav2Vec2 feature extractor and model."""

    model_path = _checkpoint_path()

    required_files = (
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
    )

    missing_files = [
        filename
        for filename in required_files
        if not (model_path / filename).is_file()
    ]

    if missing_files:
        missing = ", ".join(missing_files)
        raise ModelUnavailableError(
            f"Wav2Vec2 checkpoint is incomplete at {model_path}. "
            f"Missing: {missing}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_path,
            local_files_only=True,
        )

        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True,
        )

        model = model.to(device)
        model.eval()
    except Exception as exc:
        raise ModelUnavailableError(
            f"Wav2Vec2 checkpoint could not be loaded: {model_path}"
        ) from exc

    labels = {
        int(index): str(label).lower()
        for index, label in model.config.id2label.items()
    }

    if labels.get(0) != "bonafide" or labels.get(1) != "spoof":
        raise ModelUnavailableError(
            "Wav2Vec2 checkpoint labels must be "
            "{0: 'bonafide', 1: 'spoof'}"
        )

    return feature_extractor, model, device


def _segment_waveform(
    waveform: np.ndarray,
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    """Split a recording into four-second inference segments."""

    segments: list[np.ndarray] = []
    boundaries: list[tuple[float, float]] = []

    for start_sample in range(
        0,
        len(waveform),
        SEGMENT_SAMPLES,
    ):
        end_sample = min(
            start_sample + SEGMENT_SAMPLES,
            len(waveform),
        )

        segment = waveform[start_sample:end_sample].astype(
            np.float32,
            copy=False,
        )

        segments.append(segment)

        boundaries.append(
            (
                start_sample / SAMPLE_RATE,
                end_sample / SAMPLE_RATE,
            )
        )

    return segments, boundaries


def _predict_segment_probabilities(
    segments: list[np.ndarray],
    feature_extractor: AutoFeatureExtractor,
    model: Wav2Vec2ForSequenceClassification,
    device: torch.device,
) -> np.ndarray:
    """Run Wav2Vec2 inference over waveform segments in small batches."""

    batch_size = _inference_batch_size()
    probability_batches: list[np.ndarray] = []

    for start in range(0, len(segments), batch_size):
        waveform_batch = segments[start : start + batch_size]

        inputs = feature_extractor(
            waveform_batch,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            max_length=SEGMENT_SAMPLES,
            truncation=True,
            return_attention_mask=True,
        )

        model_inputs = {
            key: value.to(device)
            for key, value in inputs.items()
            if key in {"input_values", "attention_mask"}
        }

        with torch.inference_mode():
            logits = model(**model_inputs).logits

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

        probability_batches.append(
            probabilities.cpu().numpy()
        )

    return np.concatenate(
        probability_batches,
        axis=0,
    )


def estimate_breathing_cadence(
    waveform: np.ndarray,
) -> dict:
    """Estimate whether quiet low-frequency events follow speech endings.

    This is transparent diagnostic evidence, not proof of breathing
    and not a standalone deepfake detector.
    """

    frame_length = 1024
    hop_length = 512

    rms = librosa.feature.rms(
        y=waveform,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]

    if rms.size < 4 or float(rms.max()) <= 1e-10:
        return {
            "score": 0.0,
            "event_count": 0,
            "event_times_sec": [],
        }

    active_threshold = max(
        float(np.quantile(rms, 0.60)),
        0.10 * float(rms.max()),
    )

    active = rms >= active_threshold

    speech_end_frames = (
        np.flatnonzero(active[:-1] & ~active[1:]) + 1
    )

    stft_power = np.abs(
        librosa.stft(
            waveform,
            n_fft=N_FFT,
            hop_length=hop_length,
        )
    ) ** 2

    frequencies = librosa.fft_frequencies(
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
    )

    breathing_band = (
        (frequencies >= 100)
        & (frequencies <= 500)
    )

    event_times: list[float] = []

    look_ahead_frames = max(
        1,
        int(SAMPLE_RATE / hop_length),
    )

    for frame in speech_end_frames:
        stop = min(
            stft_power.shape[1],
            frame + look_ahead_frames,
        )

        window = stft_power[:, frame:stop]

        if window.size == 0:
            continue

        total = window.sum(axis=0)
        band = window[breathing_band, :].sum(axis=0)
        ratios = band / (total + 1e-12)
        candidate = int(np.argmax(ratios))

        if float(ratios[candidate]) >= 0.35:
            event_times.append(
                round(
                    (
                        frame
                        + candidate
                    )
                    * hop_length
                    / SAMPLE_RATE,
                    2,
                )
            )

    score = len(event_times) / max(
        1,
        len(speech_end_frames),
    )

    return {
        "score": round(
            float(np.clip(score, 0.0, 1.0)),
            4,
        ),
        "event_count": len(event_times),
        "event_times_sec": event_times,
    }


def predict_audio(audio_path: str | Path) -> dict:
    """Predict a recording using the trained Wav2Vec2 checkpoint."""

    waveform, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    if waveform.size == 0:
        raise ValueError("The uploaded audio is empty")

    feature_extractor, model, device = (
        load_prediction_model()
    )

    duration = len(waveform) / SAMPLE_RATE
    segments, boundaries = _segment_waveform(waveform)

    probabilities = _predict_segment_probabilities(
        segments,
        feature_extractor,
        model,
        device,
    )
    record_probability = np.mean(probabilities, axis=0)

    segment_results: list[dict] = []
    raw_spoof_probabilities: list[float] = []

    for boundary, probability in zip(
        boundaries,
        probabilities,
    ):
        start_sec, end_sec = boundary
        spoof_probability = float(probability[1])

        raw_spoof_probabilities.append(
            spoof_probability
        )

        segment_results.append(
            {
                "start_sec": round(start_sec, 2),
                "end_sec": round(end_sec, 2),
                "label": (
                    "spoof"
                    if spoof_probability >= SPOOF_THRESHOLD
                    else "bonafide"
                ),
                "spoof_probability": round(
                    spoof_probability,
                    6,
                ),
                "suspicious": (
                    spoof_probability >= SPOOF_THRESHOLD
                ),
            }
        )

    spoof_probability = float(
        record_probability[1]
    )

    bonafide_probability = 1.0 - spoof_probability

    predicted_label = (
        "spoof"
        if spoof_probability >= SPOOF_THRESHOLD
        else "bonafide"
    )

    confidence = max(
        spoof_probability,
        bonafide_probability,
    )

    return {
        "predicted_label": predicted_label,
        "confidence": round(confidence, 6),
        "bonafide_probability": round(
            bonafide_probability,
            6,
        ),
        "spoof_probability": round(
            spoof_probability,
            6,
        ),
        "threshold": SPOOF_THRESHOLD,
        "model_version": MODEL_VERSION,
        "duration_sec": round(duration, 2),
        "segments": segment_results,
        "breathing_cadence": (
            estimate_breathing_cadence(waveform)
        ),
    }
