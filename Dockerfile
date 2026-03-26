# Use official slim Python 3.12 image for lightweight footprint
FROM python:3.12-slim

# Prevent .pyc files and ensure logs are flush immediately (unbuffered)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set default timezone (can be overridden via ENV)
ENV TZ=America/Sao_Paulo

# Define working directory inside the container
WORKDIR /app

# Install system dependencies (build-essential for compilation, ffmpeg for Whisper)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install via pip (cache-optimized layering)
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project codebase
COPY . .

# Expose PORT injected by Google Cloud Run (default 8080)
# Use shell form to allow environment variable expansion
# 'exec' ensures signals (SIGTERM) reach the uvicorn process correctly
CMD set -e; exec uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}
