"""Typed response contracts shared by the FastAPI endpoints."""

from pydantic import BaseModel, Field


class WaveformSummary(BaseModel):
    duration_sec: float = Field(ge=0)
    peak_amplitude: float = Field(ge=0)
    rms: float = Field(ge=0)


class AnalysisResult(BaseModel):
    filename: str
    rt60_estimate_sec: float = Field(
        description="RT60 proxy in seconds; -1 means no reliable decay tail"
    )
    reverb_ratio: float = Field(ge=0)
    breathing_band_energy: float = Field(ge=0, le=1)
    mel_spectrogram_shape: list[int]
    waveform_summary: WaveformSummary


class HealthResponse(BaseModel):
    status: str
    week: int
    scope: str
