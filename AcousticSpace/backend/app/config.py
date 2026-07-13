"""Shared Week 1 configuration values."""

TARGET_SAMPLE_RATE = 16_000
MAX_AUDIO_DURATION_SECONDS = 300.0
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

ALLOWED_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a"}
ALLOWED_CONTENT_TYPES = {
    "application/octet-stream",
    "audio/flac",
    "audio/x-flac",
    "audio/aac",
    "audio/m4a",
    "audio/mp4",
    "audio/mpeg",
    "audio/wav",
    "audio/x-m4a",
    "audio/x-wav",
}
