# Use official slim Python 3.12 image for lightweight footprint
FROM python:3.12-slim

# Prevent .pyc files and ensure logs are flush immediately (unbuffered)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set default timezone
ENV TZ=America/Sao_Paulo

# Create a non-root user and group for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Define working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install via pip (cache-optimized layering)
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download FastEmbed BM25 model during build to prevent startup timeouts
RUN python3 -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding(model_name='Qdrant/bm25'); print('✅ FastEmbed BM25 model downloaded')"

# Copy the entire project codebase
COPY . .

# Change ownership of the app directory to the non-root user
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Expose PORT injected by Google Cloud Run (default 8000 if not set)
EXPOSE 8000

# Use shell form to allow environment variable expansion
# 'exec' ensures signals (SIGTERM) reach the uvicorn process correctly
CMD set -e; exec uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}
