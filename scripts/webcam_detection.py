import torch
import numpy as np
import cv2
import os

def main():
    # Load default YOLOv5s model from PyTorch Hub
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='/workspace/drowsiness_detection/model/exp/weights/best.pt', force_reload=True)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()  # Capture frame-by-frame
        if not ret:
            break

        frame = cv2.cvtColor(frame,  cv2.COLOR_BGR2RGB)
        
        # Make detections 
        results = model(frame)
        
        # Display the resulting frame
        cv2.imshow("YOLO", np.array(results.render()[0])[:, :, ::-1])

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
    

if __name__ == "__main__":
    main()