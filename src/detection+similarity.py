from ultralytics import YOLO
import cv2
import numpy as np
import os
from skimage import metrics

pothole_model = YOLO("best.pt")  #Add the path to your custom model 
general_model = YOLO("yolov8n.pt")  

save_file = "pothole_detection_images"
if not os.path.exists(save_file):
    os.makedirs(save_file)

frame_count = 0
general_classes = [0, 1, 2, 3, 5, 9, 11, 12]
cap = cv2.VideoCapture(0)

last_saved_pothole_image = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    pothole_results = pothole_model.predict(frame, conf=0.4, verbose=False)
    general_results = general_model.predict(frame, conf=0.4, show=True, verbose=False, classes=general_classes)

    pothole_frame = pothole_results[0].plot()
    general_frame = general_results[0].plot()
    combined_frame = cv2.addWeighted(pothole_frame, 0.7, general_frame, 0.7, 0)
    cv2.imshow("Combined Detection", combined_frame)

    if len(pothole_results[0].boxes) > 0:
        if last_saved_pothole_image is not None:
            last_image_gray = cv2.cvtColor(last_saved_pothole_image, cv2.COLOR_BGR2GRAY)
            current_image_gray = cv2.cvtColor(pothole_frame, cv2.COLOR_BGR2GRAY)
            current_image_gray = cv2.resize(
                current_image_gray, 
                (last_image_gray.shape[1], last_image_gray.shape[0]), 
                interpolation=cv2.INTER_AREA
            )

            ssim_score = metrics.structural_similarity(last_image_gray, current_image_gray, full=True)[0]

            if ssim_score < 0.5: #If the SSIM score will be less than 0.6 then only next image will be saved. 
                filename = f"{save_file}/pothole_{frame_count}.jpeg"
                cv2.imwrite(filename, pothole_frame)
                last_saved_pothole_image = pothole_frame.copy() 
                print("Saved", filename)
        else:
            filename = f"{save_file}/pothole_{frame_count}.jpeg"
            cv2.imwrite(filename, pothole_frame)
            last_saved_pothole_image = pothole_frame.copy() 
            print("Saved", filename)
    
    frame_count += 1

cap.release()
cv2.destroyAllWindows()
