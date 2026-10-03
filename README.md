# SMART TRAFFIC MONITORING

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow.svg)

Welcome to the **Deep Learning and Computer Vision Projects** repository! This collection features three robust, real-time computer vision applications powered by **Ultralytics YOLOv8** and **OpenCV**. These tools are engineered to autonomously track, analyze, and enforce safety compliances in dynamic real-world environments.

## Projects Overview

### 1. Vehicle Classification & Analytics
A traffic monitoring system designed to classify, track, and count vehicles using a custom-trained YOLO model and ByteTrack.

### 2. People Counting with Interactive Boundaries
A people-counting application using YOLOv8 and ByteTrack. Users define a counting boundary by clicking two points, and the system tracks entry and exit events.

### 3. Helmet Violation Detection
A safety-compliance module using a custom YOLO model to detect helmet and no-helmet classes and save cropped violation evidence.

### 4. Helper Utilities
Supporting Python utilities for dataset preparation and interface testing.

## Installation

Python 3.8+ is recommended.

```bash
pip install ultralytics
pip install opencv-python
pip install numpy
pip install PyQt5
pip install PyGetWindow
pip install mss
```

## Running the Modules

### Vehicle Analytics
```bash
cd vehicle_classification
python vehicle_classification.py
```

### People Counting
```bash
cd people_count
python people_count.py
```

### Helmet Violation Detection
```bash
cd helmet_violation
python helmet_violation.py
```

Update the video paths in the Python scripts when using your own video source. Webcam input can be configured where supported.
