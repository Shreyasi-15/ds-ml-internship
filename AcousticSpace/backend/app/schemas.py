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


class SegmentPrediction(BaseModel):
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    label: str
    spoof_probability: float = Field(ge=0, le=1)
    suspicious: bool


class BreathingCadenceResult(BaseModel):
    score: float = Field(ge=0, le=1)
    event_count: int = Field(ge=0)
    event_times_sec: list[float]


class PredictionResult(BaseModel):
    filename: str
    predicted_label: str
    confidence: float = Field(ge=0, le=1)
    bonafide_probability: float = Field(ge=0, le=1)
    spoof_probability: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    model_version: str
    duration_sec: float = Field(ge=0)
    segments: list[SegmentPrediction]
    breathing_cadence: BreathingCadenceResult


class HealthResponse(BaseModel):
    status: str
    week: int
    scope: str

class RegisterRequest(BaseModel):
    display_name: str = Field(min_length=2, max_length=60)
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=1, max_length=128)


class UserResponse(BaseModel):
    id: int
    email: str
    display_name: str
    created_at: str


class LogoutResponse(BaseModel):
    message: str

class AnalysisHistoryItem(BaseModel):
    id: int
    filename: str
    predicted_label: str
    confidence: float = Field(ge=0, le=1)
    model_version: str
    analyzed_at: str


class ModelEvaluationMetrics(BaseModel):
    accuracy: float = Field(ge=0, le=1)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1_score: float = Field(ge=0, le=1)
    eer: float = Field(ge=0, le=1)
    evaluated_recordings: int = Field(ge=0)


class ConfidenceBucket(BaseModel):
    range: str
    count: int = Field(ge=0)


class DailyPrediction(BaseModel):
    date: str
    bonafide: int = Field(ge=0)
    spoof: int = Field(ge=0)


class StatisticsResponse(BaseModel):
    total_analyses: int = Field(ge=0)
    bonafide_detections: int = Field(ge=0)
    spoof_detections: int = Field(ge=0)
    average_confidence: float = Field(ge=0, le=1)
    model_evaluation: ModelEvaluationMetrics
    confidence_distribution: list[ConfidenceBucket]
    daily_predictions: list[DailyPrediction]
