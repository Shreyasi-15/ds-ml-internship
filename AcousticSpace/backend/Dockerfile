FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ACOUSTICSPACE_MODEL_PATH=/app/models/wav2vec2_asvspoof_domain_v2/best \
    ACOUSTICSPACE_INFERENCE_BATCH_SIZE=32

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY app ./app
COPY ml ./ml
# The checkpoint is not in Git; upload it to the Space so the image carries it.
COPY models ./models

RUN useradd --create-home --uid 10001 acoustic \
    && chown -R acoustic:acoustic /app

USER acoustic

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=3)"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
