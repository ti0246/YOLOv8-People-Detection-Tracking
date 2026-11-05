# YOLOv8-Based People Detection and Multi-Object Tracking

**Undergraduate Project | Second Year**

A real-time computer vision project developed using YOLOv8, ByteTrack, and OpenCV to detect and track people in video streams.

## Overview

This project implements a real-time object detection and multi-object tracking pipeline using the YOLOv8s model.

The system processes video streams from a camera or recorded video files, detects objects, assigns tracking IDs, and monitors the number of unique person-tracking IDs observed throughout the session.

Detection results are visualized with bounding boxes, confidence scores, tracking information, and real-time FPS measurements.

## Key Features

- **Object Detection:** YOLOv8s for real-time object detection.
- **Multi-Object Tracking:** ByteTrack for tracking detected objects across video frames.
- **People Detection:** Optional person-only detection using the COCO person class.
- **Tracking ID Counting:** Tracks distinct person IDs observed during a session.
- **Performance Monitoring:** Real-time FPS display.
- **Video Processing:** Supports webcam and recorded video input.
- **Video Export:** Saves annotated video output in MP4 format.
- **Configurable Parameters:** Adjustable confidence and IoU thresholds.

## System Architecture

```text
Camera / Video Input
        |
        v
    YOLOv8s
  Object Detection
        |
        v
     ByteTrack
 Multi-Object Tracking
        |
        v
   Tracking IDs
        |
        v
 People ID Counting
        |
        v
 Visualization & FPS
        |
        v
   Video Output
```

## Technologies

| Technology | Application |
|---|---|
| Python | Main programming language |
| YOLOv8 | Object detection |
| ByteTrack | Multi-object tracking |
| OpenCV | Video processing and visualization |
| Ultralytics | YOLO model framework |
| PyTorch | Deep learning backend |

## Project Structure

```text
YOLOv8-People-Detection-Tracking/
├── DRONE.py
├── yolov8s.pt
├── people_stats.csv
├── .gitignore
└── out/
    └── result.mp4
```

## Installation

Clone the repository:

```bash
git clone https://github.com/ti0246/YOLOv8-People-Detection-Tracking.git

cd YOLOv8-People-Detection-Tracking
```

Install dependencies:

```bash
pip install ultralytics opencv-python numpy torch
```

## Usage

Run real-time detection using a webcam:

```bash
python DRONE.py --source 0 --show
```

Run detection on a video:

```bash
python DRONE.py --source input.mp4 --show
```

Detect people only:

```bash
python DRONE.py --source 0 --people-only --show
```

Save annotated output:

```bash
python DRONE.py --source input.mp4 --save out/result.mp4
```

Press `q` to exit the visualization window.

## Demo

A sample processed video is available in:

`out/result.mp4`

## Academic Background

This project was developed during my second year as an undergraduate student as part of my exploration of artificial intelligence and computer vision.

It represents an early step in my technical development toward intelligent autonomous systems, UAV applications, and robotics research.

## Author

**Vu Van Thuc**

GitHub: [@ti0246](https://github.com/ti0246)

Research Interests:
- Artificial Intelligence
- Computer Vision
- Autonomous Systems
- UAV Robotics
- Multi-Agent Systems
