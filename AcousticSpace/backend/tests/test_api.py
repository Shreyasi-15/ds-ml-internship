import io

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from app.audio_pipeline import SR
from app.main import app


client = TestClient(app)


def _wav_bytes():
    t = np.linspace(0, 1, SR, endpoint=False)
    clip = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, clip, SR, format="WAV")
    return buffer.getvalue()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_extract_features_endpoint():
    response = client.post(
        "/extract-features",
        files={"file": ("sample.wav", _wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "sample.wav"
    assert body["mel_spectrogram_shape"][0] == 64
    assert body["waveform_summary"]["duration_sec"] == 1.0


def test_invalid_extension_returns_415():
    response = client.post(
        "/extract-features",
        files={"file": ("notes.txt", b"not audio", "text/plain")},
    )
    assert response.status_code == 415


def test_empty_audio_returns_400():
    response = client.post(
        "/extract-features",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400
