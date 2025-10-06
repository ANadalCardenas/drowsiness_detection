FROM ubuntu:24.04

# Install Python, pip and git
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 python3-pip ca-certificates git \
    libgl1 libglib2.0-0 pyqt5-dev-tools

# Set the working directory
ARG WORKSPACE=/workspace/drowsiness_detection

# Create workspace directory
RUN mkdir -p $WORKSPACE
WORKDIR $WORKSPACE

# Install YOLOv5
RUN git clone https://github.com/ultralytics/yolov5.git
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN python3 -m pip install --no-cache-dir -r yolov5/requirements.txt

