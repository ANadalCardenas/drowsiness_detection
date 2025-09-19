import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2


def main():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    img = 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Cars_in_traffic_in_Auckland%2C_New_Zealand_-_copyright-free_photo_released_to_public_domain.jpg/800px-Cars_in_traffic_in_Auckland%2C_New_Zealand_-_copyright-free_photo_released_to_public_domain.jpg'
    results = model(img)
    results.print()
    img = np.squeeze(results.render())    
    plt.imshow(img)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()