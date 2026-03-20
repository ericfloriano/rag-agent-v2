# ==================================================
# Dockerfile — Agentic RAG v2 (GCP Cloud Run)
# ==================================================
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Cloud Run uses the PORT environment variable
ENV PORT=8080

# Start FastAPI with uvicorn
CMD exec uvicorn src.main:app --host 0.0.0.0 --port $PORT --workers 1
