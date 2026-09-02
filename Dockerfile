FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Install lightweight system dependencies required for OpenCV & FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to break stale Docker build cache
COPY requirements.txt .

# Install PyTorch CPU and project requirements
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY hackthon_mittul/sih26187-prototype/backend/ .

# Ensure runtime directories exist
RUN mkdir -p data/watchlist_photos data/watchlist_embeddings uploads config output

EXPOSE 8000

# Start Uvicorn with single worker and dynamic $PORT
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
