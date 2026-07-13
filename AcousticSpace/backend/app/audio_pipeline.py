"""
Audio Processing Pipeline
Isolates environmental acoustics (Room Impulse Response proxy + reverb tail)
from vocal content, and produces spectrogram features for the classifier.

Note on RIR: we don't have the *true* RIR (that requires an excitation signal
recorded in that exact room). Instead, we estimate a room-acoustics fingerprint
via:
  1. Reverberation time (RT60 estimate) from the energy decay curve
  2. Spectral decay / late-field reverb ratio
  3. Breathing-band energy (low-amplitude, low-frequency segments between speech)
These are the features a classifier can use to catch mismatches between a
voice's cadence and its claimed background environment.
"""

import librosa
import numpy as np

from app.config import MAX_AUDIO_DURATION_SECONDS, TARGET_SAMPLE_RATE

SR = TARGET_SAMPLE_RATE
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64
MAX_PLAUSIBLE_RT60_SECONDS = 10.0


class AudioValidationError(ValueError):
    """Raised when uploaded audio cannot be safely analyzed."""


def load_audio(path: str) -> np.ndarray:
    try:
        y, _ = librosa.load(path, sr=SR, mono=True)
    except Exception as exc:
        raise AudioValidationError("The uploaded file could not be decoded as audio") from exc

    if y.size == 0:
        raise AudioValidationError("The uploaded audio is empty")
    if not np.all(np.isfinite(y)):
        raise AudioValidationError("The uploaded audio contains invalid sample values")
    if len(y) / SR > MAX_AUDIO_DURATION_SECONDS:
        raise AudioValidationError(
            f"Audio must be {MAX_AUDIO_DURATION_SECONDS:.0f} seconds or shorter"
        )
    return y


def compute_mel_spectrogram(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    return librosa.power_to_db(mel, ref=np.max)


def estimate_rt60(y: np.ndarray) -> float:
    """
    Rough RT60 (reverberation time) estimate using the Schroeder backward
    integration method on the energy decay curve.
    """
    energy = y ** 2
    # Schroeder integration: reverse-cumulative sum of energy
    decay_curve = np.cumsum(energy[::-1])[::-1]
    decay_curve = decay_curve / (np.max(decay_curve) + 1e-12)
    decay_db = 10 * np.log10(decay_curve + 1e-12)

    start_candidates = np.flatnonzero(decay_db <= -5)
    end_candidates = np.flatnonzero(decay_db <= -25)
    if start_candidates.size == 0 or end_candidates.size == 0:
        return -1.0

    idx_start = int(start_candidates[0])
    idx_end = int(end_candidates[0])
    if idx_end <= idx_start:
        return -1.0

    rt60 = ((idx_end - idx_start) / SR) * 3
    return float(rt60) if 0 < rt60 <= MAX_PLAUSIBLE_RT60_SECONDS else -1.0


def estimate_reverb_ratio(y: np.ndarray) -> float:
    """
    Temporal-decay proxy. Around strong-to-weak envelope transitions, compare
    energy immediately after the transition with the direct energy before it.
    """
    frame_length = 512
    hop = 128
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]
    if len(rms) < 8 or float(np.max(rms)) <= 1e-10:
        return 0.0

    active = rms >= (0.45 * float(np.max(rms)))
    falling_edges = np.flatnonzero(active[:-1] & ~active[1:]) + 1
    direct_frames = max(2, int(0.10 * SR / hop))
    tail_frames = max(2, int(0.20 * SR / hop))
    ratios: list[float] = []

    for edge in falling_edges:
        before = rms[max(0, edge - direct_frames) : edge]
        after = rms[edge : min(len(rms), edge + tail_frames)]
        if before.size < 2 or after.size < 2:
            continue
        direct_energy = float(np.mean(before**2))
        tail_energy = float(np.mean(after**2))
        if direct_energy > 1e-12:
            ratios.append(tail_energy / direct_energy)

    if ratios:
        return float(np.median(ratios))

    high_energy = rms[active]
    low_energy = rms[~active]
    if high_energy.size == 0 or low_energy.size == 0:
        return 0.0
    return float(np.mean(low_energy**2) / (np.mean(high_energy**2) + 1e-12))


def estimate_breathing_band_energy(y: np.ndarray) -> float:
    """
    Relative 100-500 Hz energy in the quietest 30% of time frames. Keeping the
    original frame structure avoids discontinuities caused by concatenating
    unrelated silent samples. This remains a breathing proxy, not a detector.
    """
    stft = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)) ** 2
    if stft.shape[1] < 2:
        return 0.0
    frame_energy = np.mean(stft, axis=0)
    quiet_frames = frame_energy <= np.quantile(frame_energy, 0.30)
    if not quiet_frames.any():
        return 0.0

    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    band_mask = (freqs >= 100) & (freqs <= 500)
    if not band_mask.any():
        return 0.0
    quiet_power = stft[:, quiet_frames]
    total_energy = float(np.sum(quiet_power))
    if total_energy <= 1e-12:
        return 0.0
    band_energy = float(np.sum(quiet_power[band_mask, :]))
    return float(np.clip(band_energy / total_energy, 0.0, 1.0))


def extract_features(path: str) -> dict:
    """Main entry point: run the full pipeline on one audio file."""
    y = load_audio(path)

    mel_db = compute_mel_spectrogram(y)
    rt60 = estimate_rt60(y)
    reverb_ratio = estimate_reverb_ratio(y)
    breathing_energy = estimate_breathing_band_energy(y)

    duration = len(y) / SR

    return {
        "mel_spectrogram": mel_db,          # feeds the AST/CNN classifier
        "rt60_estimate": rt60,
        "reverb_ratio": reverb_ratio,
        "breathing_band_energy": breathing_energy,
        "waveform_summary": {
            "duration_sec": round(duration, 2),
            "peak_amplitude": float(np.max(np.abs(y))) if len(y) else 0.0,
            "rms": float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0,
        },
    }
