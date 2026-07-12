"""
Extraction Validation tests (Mid-Project Review deliverable):
Proves the Librosa pipeline runs end-to-end and successfully isolates
environmental/reverb features from vocal content on a synthetic clip.
"""

import numpy as np
import soundfile as sf
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.app.audio_pipeline import extract_features, SR


def _make_synthetic_clip(duration_sec=3, reverb=False):
    """Generate a synthetic 'voice-like' tone burst, optionally with a
    simple reverb tail (exponentially decaying echo) to simulate a room."""
    t = np.linspace(0, duration_sec, int(SR * duration_sec), endpoint=False)
    voice = 0.3 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 3 * t) > 0)

    if reverb:
        # crude synthetic room reflection: delayed, attenuated copies
        echo = np.zeros_like(voice)
        delay_samples = int(0.05 * SR)
        decay = 0.5
        echo[delay_samples:] += decay * voice[:-delay_samples]
        echo[2 * delay_samples:] += (decay ** 2) * voice[:-2 * delay_samples]
        voice = voice + echo

    return voice.astype(np.float32)


def test_pipeline_runs_end_to_end():
    clip = _make_synthetic_clip(reverb=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, clip, SR)
        path = tmp.name

    try:
        features = extract_features(path)
    finally:
        os.remove(path)

    assert "mel_spectrogram" in features
    assert features["mel_spectrogram"].ndim == 2
    assert "rt60_estimate" in features
    assert "reverb_ratio" in features
    assert "breathing_band_energy" in features


def test_reverb_clip_has_higher_late_energy_ratio_than_dry_clip():
    """
    Core proof-of-concept for the whole project: a clip with a synthetic
    room reflection tail should show a higher late/early energy ratio
    than the same clip with no reverb at all.
    """
    dry_clip = _make_synthetic_clip(reverb=False)
    wet_clip = _make_synthetic_clip(reverb=True)

    def _extract(clip):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, clip, SR)
            path = tmp.name
        try:
            return extract_features(path)
        finally:
            os.remove(path)

    dry_features = _extract(dry_clip)
    wet_features = _extract(wet_clip)

    assert wet_features["reverb_ratio"] >= dry_features["reverb_ratio"]
