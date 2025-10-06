import io
import requests
from PIL import Image
import torch
import numpy as np
import cv2

def main():
    # Make detectionsrivate image with my model trained on private dataset
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='/workspace/drowsiness_detection/model/exp/weights/best.pt', force_reload=True)
    img = cv2.imread("/workspace/drowsiness_detection/data/test/awake.3e3befe5-a20e-11f0-b1eb-1d37d0d6c335.jpg")
    img = cv2.cvtColor(img,  cv2.COLOR_BGR2RGB)
    results = model(img)

    # Print results to console
    results.print()

    # Show the annotated image using OpenCV
    cv2.imshow("Annotated", np.array(results.render()[0])[:, :, ::-1])
    
    # Waits for a key press to close the image window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()