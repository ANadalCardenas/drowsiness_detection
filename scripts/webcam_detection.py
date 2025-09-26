import torch
import numpy as np
import cv2
import os
import time
import uuid

def main():
    # Load default YOLOv5s model from PyTorch Hub
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

    # Initialize webcam
    """
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()  # Capture frame-by-frame
        if not ret:
            break
        
        # Make detections 
        results = model(frame)
        
        # Display the resulting frame
        cv2.imshow('YOLO', np.squeeze(results.render()))

        # Break the loop on 'q' key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
     
        # Break if window is closed manually
        try:
            if cv2.getWindowProperty('YOLO', cv2.WND_PROP_AUTOSIZE) < 0:
                break
        except cv2.error:
            break

    cap.release()
    cv2.destroyAllWindows()
    """

    # Code for collecting images from webcam for custom dataset
    IMAGES_PATH = os.path.join('data', 'images')
    # Create the directory if it doesn't exist
    os.makedirs(IMAGES_PATH, exist_ok=True)
    labels = ['rock', 'paper', 'scissors']
    number_imgs = 5
    # Loop through labels
    for label in labels:
        os.makedirs(os.path.join(IMAGES_PATH, label), exist_ok=True)
        # Create the label directory
        folder_path = os.path.join(IMAGES_PATH, label)
        cap = cv2.VideoCapture(0)
        # Loop through labels
        while cap.isOpened():        
            print('Collecting images for {}'.format(label))
            time.sleep(5)

            # Loop through image range
            for img_num in range(number_imgs):
                print('Collecting images for {}, image number {}'.format(label, img_num))

                # Webcam feed
                ret, frame = cap.read()  # Capture frame-by-frame
                if not ret:
                    break

                # Naming out image path
                imgname = os.path.join(folder_path, label+'.'+str(uuid.uuid1())+'.jpg')

                # Writes out image to file 
                cv2.imwrite(imgname, frame)

                # Render to the screen
                cv2.imshow('Image Collection', frame)

                # 2 second delay between captures
                time.sleep(2)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            # Break if window is closed manually
            try:
                if cv2.getWindowProperty('YOLO', cv2.WND_PROP_AUTOSIZE) < 0:
                    break
            except cv2.error:
                break
        cap.release()
        cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()