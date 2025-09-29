import io
import requests
from PIL import Image
import torch
import numpy as np
import cv2

def main():
    # Web image with yolov 5s model
    """
    # Load default YOLOv5s model from PyTorch Hub
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

    # Image URL (public domain image of cars in traffic)
    url = ("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/"
           "Cars_in_traffic_in_Auckland%2C_New_Zealand_-_copyright-free_photo_"
           "released_to_public_domain.jpg/800px-"
           "Cars_in_traffic_in_Auckland%2C_New_Zealand_-_copyright-free_photo_"
           "released_to_public_domain.jpg")
           
    headers = {"User-Agent": "Mozilla/5.0 (compatible; yolo-test/1.0)"}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    """
    # P
    # Make detectionsrivate image with mi model trained on rock paper scissors dataset
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='/workspace/rock_paper_scissors/yolov5/runs/train/exp/weights/best.pt', force_reload=True)
    results = model("/workspace/rock_paper_scissors/data/image.jpg")

    # Print results to console
    results.print()

    # Show the annotated image using OpenCV
    cv2.imshow("Annotated", np.array(results.render()[0])[:, :, ::-1])
    
    # Waits for a key press to close the image window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()