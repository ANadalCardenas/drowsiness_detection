# Use NVIDIA PyTorch image with CUDA support
FROM nvcr.io/nvidia/pytorch:24.04-py3

# Install system dependencies
RUN apt-get update && apt-get install -y git python3-opencv libgl1 && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Clone YOLOv5 repo
RUN git clone https://github.com/ultralytics/yolov5 /workspace/individual_predictor/yolov5

# Set working directory
WORKDIR /yolov5

# Install YOLOv5 dependencies (skip torch to avoid reinstalling)
RUN pip install -r requirements.txt --no-deps