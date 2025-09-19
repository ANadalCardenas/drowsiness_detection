# Install PyTorch (LTS 1.8 with CUDA 11.1)
RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 \
    -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

# Alternatively, install the latest PyTorch version with CUDA 11.8 support
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Clone YOLOv5
RUN git clone https://github.com/ultralytics/yolov5

# Set working directory
WORKDIR /yolov5

# Install YOLOv5 dependencies
RUN pip install -r requirements.txt