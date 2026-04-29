# AI Vehicle Detection, Tracking and Speed Estimation

## Overview
This project is an AI powered traffic analytics system that performs real time vehicle detection, tracking, speed estimation, and directional counting from video streams.
It uses **YOLOv8** for high-accuracy object detection and **DeepSORT** for robust multi-object tracking. Each vehicle is assigned a unique ID and tracked across frames, enabling speed calculation and movement direction analysis (entering vs leaving).
The system processes high-resolution video efficiently using OpenCV and generates an annotated output video with bounding boxes, vehicle IDs, speed (km/h), and live counts.
This project demonstrates practical applications of AI in **intelligent transportation systems, traffic monitoring, and real-time analytics**.

## Key Highlights
- Real time vehicle detection using deep learning (YOLOv8)
- Multi-object tracking with persistent IDs (DeepSORT)
- Speed estimation based on frame to frame motion
- Direction based vehicle counting (Entering / Leaving)
- High resolution video processing (1080p supported)
- Optimized pipeline for real time performance

## Tech Stack
- **Python**
- **OpenCV**
- **YOLOv8 (Ultralytics)**
- **DeepSORT (deep-sort-realtime)**
- **PyTorch**
- **NumPy**
  
## System Workflow
- Video input is processed frame by frame using OpenCV
- YOLOv8 detects vehicles (cars, buses, trucks, motorcycles)
- DeepSORT assigns unique IDs and tracks vehicles across frames
- Pixel displacement is used to estimate speed in km/h
- Direction is determined based on line crossing logic
- Output video is generated with annotations and live metrics   

## How to Run
- Install dependencies:
```bash
pip install -r requirements.txt
python main.py
