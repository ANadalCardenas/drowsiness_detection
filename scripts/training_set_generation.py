import torch
import numpy as np
import cv2
import os
import time
import uuid
import shutil

LABELS = ['rock', 'paper', 'scissors']
# current script folder = /workspace/rock_paper_scissors/scripts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # goes one level up
IMAGES_PATH = os.path.join(PROJECT_ROOT, "data", "images")
# number of images to collect per label
NUM_IMGS = 5 

def main():
    for label in LABELS:
        print(f"Starting collection for label: {label}")
        os.makedirs(IMAGES_PATH, exist_ok=True)

        # Open webcam for this label
        cap = cv2.VideoCapture(0)
        time.sleep(2)  # warm-up time

        img_count = 0
        while img_count < NUM_IMGS:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Failed to capture image")
                break

            # Show live webcam feed
            cv2.imshow(f'Image Collection: {label}', frame)

            # Capture on spacebar, quit on q
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # spacebar = save
                imgname = os.path.join(IMAGES_PATH, f"{label}.{uuid.uuid1()}.jpg")
                cv2.imwrite(imgname, frame)
                img_count += 1
                print(f"✅ Saved image {img_count} for {label}")
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        cap.release()
        cv2.destroyAllWindows()
        print(f"Finished collecting {img_count} images for {label}")
        time.sleep(2)  # pause between labels

    print("Image collection completed ✅")

    # Get the parent folder of images (i.e., 'data')
    base_dir = os.path.dirname(IMAGES_PATH)
    # Create 'labels' folder at the same level
    LABELS_PATH = os.path.join(base_dir, "labels")
    os.makedirs(LABELS_PATH, exist_ok=True)


if __name__ == "__main__":
    main()