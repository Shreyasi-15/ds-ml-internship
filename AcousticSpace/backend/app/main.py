"""
AcousticSpace API Gateway — Week 1 scope.

Week 1 goal (per project plan): "Build FastAPI server. Curate dataset.
Develop Librosa pipeline to extract spectrograms and RIR features."

This server does NOT classify audio yet — that's Week 2, once the
PyTorch baseline model exists. Right now it proves the pipeline works
end-to-end: upload audio -> get back extracted acoustic features.

Run locally:
    uvicorn app.main:app --reload --port 8000

Then test at: http://localhost:8000/docs (FastAPI's built-in test UI)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import tempfile
import os

from app.audio_pipeline import extract_features

app = FastAPI(
    title="AcousticSpace API (Week 1)",
    description="Audio feature extraction pipeline — RIR, reverb, and spectrogram analysis",
    version="0.1.0",
)

# Allows your React dashboard (localhost:5173) to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """Friendly root response so opening port 8000 does not show 404."""
    return {
        "message": "AcousticSpace API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health_check():
    """Quick check that the server is alive."""
    return {"status": "ok", "week": 1, "scope": "feature extraction only, no classifier yet"}


@app.post("/extract-features")
async def extract_features_endpoint(file: UploadFile = File(...)):
    """
    Accepts an audio file, runs the Librosa pipeline, and returns the
    raw acoustic features (RT60 estimate, reverb ratio, breathing-band
    energy, spectrogram shape). No deepfake verdict yet — that requires
    the Week 2 classifier.
    """
    if not file.filename.lower().endswith((".wav", ".flac", ".mp3", ".m4a")):
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        features = extract_features(tmp_path)
    finally:
        os.remove(tmp_path)

    return {
        "filename": file.filename,
        "rt60_estimate_sec": features["rt60_estimate"],
        "reverb_ratio": features["reverb_ratio"],
        "breathing_band_energy": features["breathing_band_energy"],
        "mel_spectrogram_shape": list(features["mel_spectrogram"].shape),
        "waveform_summary": features["waveform_summary"],
    }
