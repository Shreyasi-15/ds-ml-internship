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

import numpy as np
import librosa

SR = 16000            # target sample rate
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64
MAX_PLAUSIBLE_RT60_SECONDS = 10.0


def load_audio(path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=True)
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

    # Find time to decay from -5dB to -25dB, then extrapolate to -60dB (RT20 method)
    try:
        idx_start = np.argmax(decay_db <= -5)
        idx_end = np.argmax(decay_db <= -25)
        if idx_end <= idx_start:
            return -1.0  # not enough decay range in this clip
        t_start, t_end = idx_start / SR, idx_end / SR
        rt20 = t_end - t_start
        rt60 = rt20 * 3  # extrapolate RT20 -> RT60
        # Sustained speech/music is not a free decay tail; do not report an
        # implausible RT60 calculated from the full recording duration.
        return float(rt60) if 0 < rt60 <= MAX_PLAUSIBLE_RT60_SECONDS else -1.0
    except Exception:
        return -1.0


def estimate_reverb_ratio(y: np.ndarray) -> float:
    """
    Late-field vs early-field energy ratio — a proxy for how much of the
    signal's tail comes from room reflections vs the direct/dry voice.
    """
    frame_length = 2048
    hop = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]
    if len(rms) < 4:
        return 0.0
    early = rms[: len(rms) // 3]
    late = rms[len(rms) // 3 :]
    early_energy = np.mean(early ** 2) + 1e-12
    late_energy = np.mean(late ** 2) + 1e-12
    return float(late_energy / early_energy)


def estimate_breathing_band_energy(y: np.ndarray) -> float:
    """
    Breathing sits mostly in low-amplitude, low-frequency segments between
    voiced speech. We isolate non-voiced low-energy frames and measure their
    spectral energy in the 100-500 Hz band as a breathing-pattern proxy.
    """
    intervals = librosa.effects.split(y, top_db=30)
    non_speech_mask = np.ones(len(y), dtype=bool)
    for start, end in intervals:
        non_speech_mask[start:end] = False
    non_speech = y[non_speech_mask]

    if len(non_speech) < SR // 10:  # too little non-speech audio to trust
        return 0.0

    stft = np.abs(librosa.stft(non_speech, n_fft=N_FFT, hop_length=HOP_LENGTH))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    band_mask = (freqs >= 100) & (freqs <= 500)
    band_energy = np.mean(stft[band_mask, :]) if band_mask.any() else 0.0
    return float(band_energy)


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
