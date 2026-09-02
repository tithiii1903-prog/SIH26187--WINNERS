FROM python:3.10-slim

# Prevent Python from writing .pyc files & enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Install system dependencies required for OpenCV, PyTorch, InsightFace & ONNX
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model weights
COPY . .

# Ensure runtime directories exist
RUN mkdir -p data/watchlist_photos data/watchlist_embeddings uploads config output

EXPOSE 8000

# Start Uvicorn with support for cloud dynamic $PORT (e.g. Render / Railway / Cloud Run)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
