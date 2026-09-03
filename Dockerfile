FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PORT=8080
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

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

# Upgrade pip, setuptools and wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Pin numpy==1.26.4 FIRST to prevent NumPy 2.0 ONNX/ml_dtypes incompatibilities
RUN pip install --no-cache-dir "numpy==1.26.4"

# Install CPU PyTorch from official CPU wheel repository
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install remaining packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY hackthon_mittul/sih26187-prototype/backend/ .

# Ensure runtime directories exist
RUN mkdir -p data/watchlist_photos data/watchlist_embeddings uploads config output

EXPOSE 8080

# Start Uvicorn with single worker and dynamic $PORT
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
