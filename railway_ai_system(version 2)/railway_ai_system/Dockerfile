# ──────────────────────────────────────────────────────────────────────────────
# RailDrishti AI — Edge Node Dockerfile
# Runs: YOLOv8-Pose inference + FastAPI + OpenCV on station-level hardware
# Target: NVIDIA Jetson / x86 Industrial PC with camera attached
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ ./api/
COPY frontend/dist/ ./frontend/dist/

# Pre-download the YOLOv8-pose model weights (avoids download on first run)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"

# Environment configuration
ENV RAILDRISHTI_DEMO=true
ENV PYTHONUNBUFFERED=1

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD wget -qO- http://localhost:8000/api/system_health || exit 1

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
