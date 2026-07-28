"""API contract tests for the Week 3 prediction endpoint."""

import io

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

import app.main as main_module
from app.audio_pipeline import SR


client = TestClient(main_module.app)


def _wav_bytes() -> bytes:
    time = np.linspace(0, 1, SR, endpoint=False)
    clip = (0.2 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, clip, SR, format="WAV")
    return buffer.getvalue()


def test_predict_endpoint_returns_confidence_and_segments(monkeypatch):
    def fake_prediction(_path):
        return {
            "predicted_label": "spoof",
            "confidence": 0.82,
            "bonafide_probability": 0.18,
            "spoof_probability": 0.82,
            "threshold": 0.50,
            "model_version": "baseline-cnn-v1",
            "duration_sec": 4.0,
            "segments": [
                {
                    "start_sec": 0.0,
                    "end_sec": 4.0,
                    "label": "spoof",
                    "spoof_probability": 0.82,
                    "suspicious": True,
                }
            ],
            "breathing_cadence": {
                "score": 0.5,
                "event_count": 1,
                "event_times_sec": [2.1],
            },
        }

    monkeypatch.setattr(main_module, "predict_audio", fake_prediction)
    response = client.post(
        "/predict",
        files={"file": ("sample.wav", _wav_bytes(), "audio/wav")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "sample.wav"
    assert body["predicted_label"] == "spoof"
    assert body["spoof_probability"] == 0.82
    assert body["segments"][0]["suspicious"] is True
    assert body["breathing_cadence"]["event_count"] == 1
