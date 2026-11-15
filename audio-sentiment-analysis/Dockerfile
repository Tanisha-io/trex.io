FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WHISPER_MODEL=base \
    WHISPER_BACKEND=faster \
    WHISPER_DEVICE=cpu \
    WHISPER_COMPUTE_TYPE=int8 \
    PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Weights: either bake models_final into the image, or download at start from MODEL_REPO.
RUN pip install --no-cache-dir "huggingface_hub>=0.23" \
    && chmod +x entrypoint.sh

EXPOSE 7860

ENV MODEL_REPO=qetzi/voice-emotion-models
CMD ["./entrypoint.sh"]
