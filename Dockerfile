# Use NVIDIA PyTorch image with CUDA support
FROM nvcr.io/nvidia/pytorch:24.04-py3

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && \
    apt-get install -y --no-install-recommends git python3-opencv libgl1 tzdata && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Create workspace directory
RUN mkdir -p /workspace/rock_paper_scissors
WORKDIR /workspace/rock_paper_scissors

# Clone YOLOv5 repo
RUN git clone https://github.com/ultralytics/yolov5.git

# Install YOLOv5 dependencies (remove torch from requirements to avoid reinstalling it)
RUN cd yolov5 && sed -i '/^torch/Id' requirements.txt && \
    python -m pip install --no-cache-dir -r requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir torch