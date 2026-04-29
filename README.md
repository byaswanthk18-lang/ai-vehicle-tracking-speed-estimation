# AI Vehicle Detection, Tracking and Speed Estimation

## Project Overview
This project detects vehicles using YOLO, tracks them using DeepSORT, and estimates vehicle speed from video frames. It also counts incoming and outgoing vehicles.

## Technologies Used
- Python
- OpenCV
- YOLOv8 (Ultralytics)
- DeepSORT
- PyTorch
- NumPy

## Features
- Vehicle detection using YOLOv8
- Vehicle tracking using DeepSORT
- Speed estimation in km/h
- Entering and leaving vehicle count
- Output video generation

## How to Run
1. Install dependencies:
```bash
pip install -r requirements.txt
python main.py