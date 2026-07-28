"""Week 3 inference for the trained AcousticSpace baseline CNN."""

from functools import lru_cache
import os
from pathlib import Path

import librosa
import numpy as np
import torch

from ml.model import BaselineCNN


BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = BACKEND_ROOT / "models" / "baseline_cnn.pt"

SAMPLE_RATE = 16_000
SEGMENT_SECONDS = 4.0
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_SECONDS)
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64
SPOOF_THRESHOLD = 0.50


class ModelUnavailableError(RuntimeError):
    """Raised when the local trained checkpoint cannot be used."""


def _checkpoint_path() -> Path:
    configured = os.getenv("ACOUSTICSPACE_MODEL_PATH")
    return Path(configured).expanduser() if configured else DEFAULT_MODEL_PATH


@lru_cache(maxsize=1)
def load_prediction_model() -> tuple[BaselineCNN, dict, torch.device]:
    """Load and cache the trained CNN checkpoint."""

    model_path = _checkpoint_path()
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise ModelUnavailableError(
            f"Model checkpoint is missing or empty: {model_path}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(
            model_path,
            map_location=device,
            weights_only=True,
        )
        model = BaselineCNN().to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
    except Exception as exc:
        raise ModelUnavailableError(
            f"Model checkpoint could not be loaded: {model_path}"
        ) from exc

    return model, checkpoint, device


def waveform_to_log_mel(waveform: np.ndarray) -> torch.Tensor:
    """Convert one four-second waveform into the CNN input format."""

    waveform = librosa.util.fix_length(
        waveform.astype(np.float32),
        size=SEGMENT_SAMPLES,
    )
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    mean = float(log_mel.mean())
    standard_deviation = float(log_mel.std())
    if standard_deviation > 1e-6:
        log_mel = (log_mel - mean) / standard_deviation
    else:
        log_mel = log_mel - mean

    return torch.from_numpy(log_mel).float().unsqueeze(0)


def estimate_breathing_cadence(waveform: np.ndarray) -> dict:
    """Estimate whether low-frequency quiet events follow speech endings.

    This is transparent diagnostic evidence, not a breathing detector.
    """

    frame_length = 1024
    hop_length = 512
    rms = librosa.feature.rms(
        y=waveform,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]
    if rms.size < 4 or float(rms.max()) <= 1e-10:
        return {"score": 0.0, "event_count": 0, "event_times_sec": []}

    active_threshold = max(
        float(np.quantile(rms, 0.60)),
        0.10 * float(rms.max()),
    )
    active = rms >= active_threshold
    speech_end_frames = np.flatnonzero(active[:-1] & ~active[1:]) + 1

    stft_power = np.abs(
        librosa.stft(
            waveform,
            n_fft=N_FFT,
            hop_length=hop_length,
        )
    ) ** 2
    frequencies = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)
    breathing_band = (frequencies >= 100) & (frequencies <= 500)

    event_times: list[float] = []
    look_ahead_frames = max(1, int(SAMPLE_RATE / hop_length))
    for frame in speech_end_frames:
        stop = min(stft_power.shape[1], frame + look_ahead_frames)
        window = stft_power[:, frame:stop]
        if window.size == 0:
            continue
        total = window.sum(axis=0)
        band = window[breathing_band, :].sum(axis=0)
        ratios = band / (total + 1e-12)
        candidate = int(np.argmax(ratios))
        if float(ratios[candidate]) >= 0.35:
            event_times.append(
                round((frame + candidate) * hop_length / SAMPLE_RATE, 2)
            )

    score = len(event_times) / max(1, len(speech_end_frames))
    return {
        "score": round(float(np.clip(score, 0.0, 1.0)), 4),
        "event_count": len(event_times),
        "event_times_sec": event_times,
    }


def predict_audio(audio_path: str | Path) -> dict:
    """Predict the whole recording and score each four-second segment."""

    waveform, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    if waveform.size == 0:
        raise ValueError("The uploaded audio is empty")

    model, checkpoint, device = load_prediction_model()
    duration = len(waveform) / SAMPLE_RATE
    segment_results: list[dict] = []

    tensors: list[torch.Tensor] = []
    boundaries: list[tuple[float, float]] = []
    for start_sample in range(0, len(waveform), SEGMENT_SAMPLES):
        end_sample = min(start_sample + SEGMENT_SAMPLES, len(waveform))
        tensors.append(
            waveform_to_log_mel(waveform[start_sample:end_sample])
        )
        boundaries.append(
            (
                start_sample / SAMPLE_RATE,
                end_sample / SAMPLE_RATE,
            )
        )

    batch = torch.stack(tensors).to(device)
    with torch.inference_mode():
        probabilities = model.predict_probabilities(batch).cpu().numpy()

    for (start_sec, end_sec), probability in zip(boundaries, probabilities):
        spoof_probability = float(probability[1])
        segment_results.append(
            {
                "start_sec": round(start_sec, 2),
                "end_sec": round(end_sec, 2),
                "label": (
                    "spoof"
                    if spoof_probability >= SPOOF_THRESHOLD
                    else "bonafide"
                ),
                "spoof_probability": round(spoof_probability, 6),
                "suspicious": spoof_probability >= SPOOF_THRESHOLD,
            }
        )

    spoof_probability = float(
        np.mean([item["spoof_probability"] for item in segment_results])
    )
    bonafide_probability = 1.0 - spoof_probability
    predicted_label = (
        "spoof" if spoof_probability >= SPOOF_THRESHOLD else "bonafide"
    )
    confidence = max(spoof_probability, bonafide_probability)

    return {
        "predicted_label": predicted_label,
        "confidence": round(confidence, 6),
        "bonafide_probability": round(bonafide_probability, 6),
        "spoof_probability": round(spoof_probability, 6),
        "threshold": SPOOF_THRESHOLD,
        "model_version": checkpoint.get(
            "model_version",
            "baseline-cnn-v1",
        ),
        "duration_sec": round(duration, 2),
        "segments": segment_results,
        "breathing_cadence": estimate_breathing_cadence(waveform),
    }
