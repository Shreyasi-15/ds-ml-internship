"""AcousticSpace FastAPI gateway for acoustic analysis and prediction."""

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from app.audio_pipeline import AudioValidationError, extract_features
from app.config import ALLOWED_CONTENT_TYPES, ALLOWED_EXTENSIONS, MAX_UPLOAD_BYTES
from app.inference import ModelUnavailableError, predict_audio
from app.schemas import AnalysisResult, HealthResponse, PredictionResult


app = FastAPI(
    title="AcousticSpace API",
    description="Acoustic feature extraction and deepfake-audio prediction",
    version="0.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "AcousticSpace API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "ok",
        "week": 4,
        "scope": (
            "trained AST classification, acoustic evidence, "
            "breathing-cadence diagnostics, suspicious segments, "
            "and Docker deployment"
        ),
    }


async def _read_validated_upload(file: UploadFile) -> tuple[str, str, bytes]:
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="A filename is required")

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported audio format. Allowed extensions: {allowed}",
        )
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type: {file.content_type}",
        )

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        )
    return filename, suffix, content


def _write_temporary_audio(content: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
        temporary.write(content)
        return temporary.name


@app.post("/extract-features", response_model=AnalysisResult)
async def extract_features_endpoint(file: UploadFile = File(...)):
    filename, suffix, content = await _read_validated_upload(file)
    tmp_path: str | None = None
    try:
        tmp_path = _write_temporary_audio(content, suffix)
        features = await run_in_threadpool(extract_features, tmp_path)
    except AudioValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {
        "filename": filename,
        "rt60_estimate_sec": features["rt60_estimate"],
        "reverb_ratio": features["reverb_ratio"],
        "breathing_band_energy": features["breathing_band_energy"],
        "mel_spectrogram_shape": list(features["mel_spectrogram"].shape),
        "waveform_summary": features["waveform_summary"],
    }


@app.post("/predict", response_model=PredictionResult)
async def predict_endpoint(file: UploadFile = File(...)):
    filename, suffix, content = await _read_validated_upload(file)
    tmp_path: str | None = None
    try:
        tmp_path = _write_temporary_audio(content, suffix)
        prediction = await run_in_threadpool(predict_audio, tmp_path)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (AudioValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {"filename": filename, **prediction}
