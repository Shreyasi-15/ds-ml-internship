"""Controlled validation tests for AcousticSpace acoustic proxies."""

import numpy as np

from app.audio_pipeline import (
    SR,
    estimate_breathing_band_energy,
    estimate_rt60,
)


def create_exponential_rir(
    rt60_seconds: float,
    duration_seconds: float = 2.0,
) -> np.ndarray:
    """
    Create a controlled exponentially decaying impulse response.

    The amplitude reaches -60 dB at approximately rt60_seconds.
    """

    time = np.arange(
        int(SR * duration_seconds),
        dtype=np.float32,
    ) / SR

    decay_rate = 6.9078 / rt60_seconds
    rir = np.exp(-decay_rate * time)

    return rir.astype(np.float32)


def create_speech_with_quiet_band(
    quiet_frequency: float,
) -> np.ndarray:
    """Create loud speech-like audio followed by a quiet frequency band."""

    active_duration = 1.0
    quiet_duration = 3.0

    active_time = np.arange(
        int(SR * active_duration),
        dtype=np.float32,
    ) / SR

    quiet_time = np.arange(
        int(SR * quiet_duration),
        dtype=np.float32,
    ) / SR

    # A louder harmonic signal acts as the vocal portion.
    active_signal = (
        0.40 * np.sin(2 * np.pi * 180 * active_time)
        + 0.20 * np.sin(2 * np.pi * 360 * active_time)
        + 0.10 * np.sin(2 * np.pi * 720 * active_time)
    )

    # A quieter signal represents audio between speech segments.
    quiet_signal = (
        0.025
        * np.sin(2 * np.pi * quiet_frequency * quiet_time)
    )

    return np.concatenate(
        [active_signal, quiet_signal],
    ).astype(np.float32)


def test_rt60_proxy_tracks_known_room_decay():
    """A longer artificial room decay should produce a longer RT60."""

    small_room_rir = create_exponential_rir(0.30)
    large_room_rir = create_exponential_rir(0.90)

    small_room_rt60 = estimate_rt60(small_room_rir)
    large_room_rt60 = estimate_rt60(large_room_rir)

    assert 0.25 <= small_room_rt60 <= 0.35
    assert 0.80 <= large_room_rt60 <= 1.00
    assert large_room_rt60 > small_room_rt60


def test_breathing_proxy_focuses_on_quiet_low_frequency_audio():
    """
    Quiet 250 Hz energy should produce a stronger breathing-band
    measurement than quiet 2000 Hz energy.
    """

    breathing_band_audio = create_speech_with_quiet_band(250)
    outside_band_audio = create_speech_with_quiet_band(2000)

    breathing_ratio = estimate_breathing_band_energy(
        breathing_band_audio,
    )
    outside_band_ratio = estimate_breathing_band_energy(
        outside_band_audio,
    )

    assert breathing_ratio > outside_band_ratio
    assert breathing_ratio > 0.50
    assert outside_band_ratio < 0.10
