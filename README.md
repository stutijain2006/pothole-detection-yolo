# pothole-detection-yolo
Real-time pothole detection system using a custom YOLOv8 model trained on Roboflow and deployed with OpenCV. The system captures webcam feed, detects potholes with bounding boxes, and uses SSIM similarity filtering to avoid duplicate image logging. Includes Colab notebook for model training, best.pt weights, and Python inference scripts.

This project implements a **real-time pothole detection system** using a **custom-trained YOLOv8 model**.  
The system captures webcam video, detects potholes, draws bounding boxes, and automatically saves **only unique pothole frames** using SSIM similarity filtering.

🚧 Designed for smart city applications, road-safety automation & AI-powered infrastructure monitoring.

---

## 📌 Features

- ✅ Custom YOLOv8 object detection model trained on Roboflow
- ✅ Live webcam detection (OpenCV)
- ✅ Bounding box and confidence display
- ✅ Saves pothole images automatically
- ✅ SSIM image similarity check to avoid duplicate saves
- ✅ Public dataset – easily retrainable
- ✅ Google Colab notebook provided for full reproducibility
- ✅ MIRA SDK integration ready for cloud post-processing

---

## 🧠 Model & Dataset

- **Dataset Source**: Custom pothole dataset collected & labeled on Roboflow
- **Model**: YOLOv8
- **Training Environment**: Google Colab (GPU)

### 🔗 Public Dataset Access
This model was trained on a **public Roboflow dataset**.
To download it in Colab / Python, enter your own Roboflow API key:

```python
from roboflow import Roboflow
rf = Roboflow(api_key=input("Enter your Roboflow API Key: "))
project = rf.workspace("stuti-jain").project("pothole-detector-pm96b")
version = project.version(1)
dataset = version.download("yolov8")
Get your API key: https://roboflow.com
```

## 🚀 Running Real-Time Detection
✅ Install Dependencies
  ``` pip install ultralytics opencv-python scikit-image numpy ```
✅ Run Detection Script
  ``` python src/detect_potholes.py ```
  Press q to quit the webcam stream.

## Training the Model (Colab)
Open the included notebook:
```notebooks/Pothole_Detector.ipynb```


## Steps:
1. Upload notebook to Colab
2. Enter your Roboflow API Key when prompted
3. Train YOLOv8
4. Download best.pt weights
5. Update the path of your best.pt file to detection+similarity.py file. 

## Applications
1. Smart Road Safety Systems
2. Autonomous Vehicle Road Monitoring
3. Municipal Maintenance Automation
4. Civil Engineering & Infrastructure AI