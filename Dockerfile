# Use PyTorch CUDA runtime base image
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# Set environment variables
ENV HF_HOME=/data/huggingface
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ca-certificates \
    curl \
    ffmpeg \
    libgl1 \
    libjpeg-turbo8 \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

# Create cache directories
RUN mkdir -p /data/huggingface

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir \
    runpod \
    huggingface-hub \
    hf-transfer \
    transformers \
    diffusers>=0.35.1 \
    accelerate \
    safetensors \
    timm \
    einops \
    open_clip_torch \
    Pillow \
    numpy \
    minio \
    requests \
    aiohttp \
    pydantic \
    python-dotenv \
    uvloop \
    opencv-python-headless

# Copy handler
COPY handler.py .

# Import smoke test
RUN python -c "import runpod, transformers, diffusers, torch, PIL, minio, requests, pydantic; from diffusers import QwenImageEditPipeline; print('All imports successful')"

# Set entrypoint
CMD ["python", "-u", "handler.py"]