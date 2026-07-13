"""Validation tests for the Week 1 acoustic feature pipeline."""

import os
import tempfile

import numpy as np
import soundfile as sf

from app.audio_pipeline import N_MELS, SR, extract_features


def _make_synthetic_clip(duration_sec=3.0, reverb=False):
    t = np.linspace(0, duration_sec, int(SR * duration_sec), endpoint=False)
    gate = ((t % 0.5) < 0.18).astype(np.float32)
    voice = (0.3 * np.sin(2 * np.pi * 220 * t) * gate).astype(np.float32)

    if reverb:
        impulse = np.zeros(int(0.25 * SR), dtype=np.float32)
        impulse[0] = 1.0
        impulse[int(0.05 * SR)] = 0.45
        impulse[int(0.10 * SR)] = 0.25
        voice = np.convolve(voice, impulse, mode="full")[: len(voice)].astype(np.float32)
    return voice


def _extract(clip):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
        sf.write(temporary.name, clip, SR)
        path = temporary.name
    try:
        return extract_features(path)
    finally:
        os.remove(path)


def test_pipeline_returns_finite_typed_features():
    features = _extract(_make_synthetic_clip(reverb=True))

    assert features["mel_spectrogram"].ndim == 2
    assert features["mel_spectrogram"].shape[0] == N_MELS
    assert np.all(np.isfinite(features["mel_spectrogram"]))
    assert np.isfinite(features["rt60_estimate"])
    assert features["reverb_ratio"] >= 0
    assert 0 <= features["breathing_band_energy"] <= 1
    assert features["waveform_summary"]["duration_sec"] == 3.0


def test_reverb_clip_has_meaningfully_higher_decay_ratio():
    dry = _extract(_make_synthetic_clip(reverb=False))
    wet = _extract(_make_synthetic_clip(reverb=True))

    assert wet["reverb_ratio"] > dry["reverb_ratio"] + 1e-4
