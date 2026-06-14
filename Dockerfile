FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Prevent any interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Kolkata

WORKDIR /app

# Install system dependencies demucs needs
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your server code
COPY main.py .

# RunPod handler wrapper
COPY handler.py .

CMD ["python", "handler.py"]