FROM ubuntu:24.04

# Install Python, pip and git
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 python3-pip ca-certificates git

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
# Patch labelImg (copy small script and run it)
COPY patch_labelimg.py /tmp/patch_labelimg.py
RUN python3 /tmp/patch_labelimg.py && rm -f /tmp/patch_labelimg.py
