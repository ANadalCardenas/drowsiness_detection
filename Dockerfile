# Use NVIDIA PyTorch image with CUDA support
FROM nvcr.io/nvidia/pytorch:24.04-py3

# System packages for Tk (simplest GUI backend) and X11
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-tk tk xauth x11-apps git \
 && rm -rf /var/lib/apt/lists/*

# Create workspace directory
RUN mkdir -p /workspace/rock_paper_scissors
WORKDIR /workspace/rock_paper_scissors

# Clone YOLOv5 repo
RUN git clone https://github.com/ultralytics/yolov5.git

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install YOLOv5 dependencies (remove torch from requirements to avoid reinstalling it)
RUN cd yolov5 && sed -i '/^torch/Id' requirements.txt && \
    python -m pip install --no-cache-dir -r requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir matplotlib numpy opencv-python