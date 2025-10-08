# Drowsiness Detection using YOLOv5

## 🧠 Introduction

This project implements a **drowsiness detection system** using a **YOLOv5 deep learning model**.  
The system detects whether a person appears **awake or drowsy** from webcam or image input.  

It includes scripts for:
- Collecting and labeling your own dataset.
- Training a YOLOv5 model on your data.
- Running inference using either webcam or images.

### 🎥 Demo Video

The full demonstration video is available to view by clicking on the next GIF:

[![Watch the demo](media/demo_preview.gif)](media/demo.webm)


---

## ⚙️ Installation

Follow these steps to install all dependencies and set up the Docker environment:

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/drowsiness_detection.git
cd drowsiness_detection
```

### 2. Build the Docker image
```bash
docker compose build yolov5
```

### 3. Run the Docker container
```bash
docker compose up yolov5
```

### 4. Access the container shell
```bash
docker exec -it yolov5_container bash
```

This environment includes:
- Python 3.10+
- PyTorch
- OpenCV
- YOLOv5
- All required system dependencies

---

## 🚀 Execution

Once inside the container (`yolov5_container`), you can run the following scripts:

### Run image detection
```bash
python3 scripts/image_detection.py
```

### Run webcam detection
```bash
python3 scripts/webcam_detection.py
```

Both scripts load your trained YOLOv5 model from:
```
/workspace/drowsiness_detection/model/exp/weights/best.pt
```

The webcam version will open a live feed and perform real-time drowsiness detection.
Press **'q'** to quit.

---

## 🧩 Training Pipeline

Follow these steps to **collect, label, and train your model**:

### 1. Image collection
Run the following script to collect images for training:
```bash
python3 scripts/training_set_generation.py
```

This will:
- Capture images for each defined label (by default: *awake* and *drowsy*).
- Save them into the folder `data/images`.

### 2. Labeling
Use **Label Studio** to annotate your collected images.  
Export labels into YOLO Image format and place them into the folder:
```
data/labels/
```

Each `.txt` file must correspond to one image with the same name.

### 3. Dataset configuration
The dataset is defined in `data/data_set.yaml`:
```yaml
path: /workspace/drowsiness_detection/data
train: images
val: images
names:
  0: awake
  1: drowsy
```
The order in which the labels appear is very important. The order of the labels/classes.txt must be the same.

### 4. Train your model
Run the training script inside the container:
```bash
bash train.sh
```

This script calls YOLOv5’s training module using your dataset. Once training, save the trained weights from 
```
yolov5/runs/train/exp
```
in:
```
data/model/exp/
```

### 5. Test your model
Once training is complete, you can use:
```bash
python3 scripts/image_detection.py
```
or
```bash
python3 scripts/webcam_detection.py
```
to verify the model’s performance.

---

## 📁 Project Structure

```
drowsiness_detection/
├── scripts/
│   ├── image_detection.py
│   ├── webcam_detection.py
│   └── training_set_generation.py
├── data/
│   ├── images/
│   ├── labels/
│   ├── model/exp/
│   └── data_set.yaml
├── Dockerfile
├── docker_compose.yaml
└── train.sh
```

---

## 🧰 Tools Used

- **YOLOv5** (object detection model)
- **PyTorch** (deep learning framework)
- **OpenCV** (image processing)
- **Docker** (environment management)
- **Label Studio** (annotation tool)

---

## 🧑‍💻 Author

Developed by **Aina Nadal**  
Mathematician & Computer Scientist specialized in AI and data analysis.

---

## 📜 License

This project is released under the MIT License.  
Feel free to use, modify, and distribute it for educational and research purposes.