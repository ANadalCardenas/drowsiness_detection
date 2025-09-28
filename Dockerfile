FROM ubuntu:24.04

# System packages for Tk (simplest GUI backend) and X11
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 python3-pip python3-tk tk xauth x11-apps \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1

# Additional dependencies
RUN apt-get install -y git python3 python3-pip ca-certificates

# Set the working directory
ARG WORKSPACE=/workspace/rock_paper_scissors

# Create workspace directory
RUN mkdir -p $WORKSPACE
WORKDIR $WORKSPACE

# Install YOLOv5
RUN git clone https://github.com/ultralytics/yolov5.git
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN python3 -m pip install --no-cache-dir -r yolov5/requirements.txt

# Install labelImg
RUN apt-get install -y pyqt5-dev-tools
RUN python3 -m pip install --no-cache-dir labelImg
