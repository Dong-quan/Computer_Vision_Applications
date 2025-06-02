# Computer Vision Application (Image Processing)

**Course Project – Computer Vision with Python**

### Authors
- **Huynh Vo Phuc Loc** – 22146344  
- **Nguyen Thien Nhan** – 22146364  

---

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Installation Guide](#installation-guide) 
4. [Usage](#usage)
5. [Troubleshooting](#troubleshooting)
6. [Author](#author)

---

## Overview

This is a Computer Vision application project built using **Streamlit** that includes various image processing functions and recognition modules. The project was developed for the Computer Vision course in the Mechatronics Engineering Technology program at Ho Chi Minh City University of Technology and Education.

---

## Features

This project provides 7 main features:

- **Advanced image processing tools** from chapters 3, 4, and 9 of the book *Digital Image Processing 2018 Local Version*, including:  
 `Negative`, `Logarithmic`, `Negative Color`, `Power`, `Piecewise Line`, `Histogram`, `Histequal`, `Local Hist`, `Hist Stat`, `Smooth Box`, `Smooth Gauss`, `Median Filter`, `Sharpen`, `Gradient`, `Spectrum`, `Draw Notch Filter`, `Remove Moire Simple`, `Remove Periodic Noise`, `Draw Periodic Notch Filter`, `Erosion`, `Dilation`, `Boundary`, `Contour`, `ConvexHull`, `Defect Detection`, `Hole Fill`, `Connected Components`, `Remove Small Objects`.

- **Face Recognition**: Detect and recognize 5 specific faces (`Loc`, `Nhan`, `Loi`, `Phong`, `Liem`) via webcam or video stream. Unknown faces will be labeled as "unknown".

- **Shape Detection**: Identify geometric shapes (square, circle, triangle) from a given image set.

- **Material Recognition**: Classify materials such as wood, fabric, and metal.

- **Fruit Detection and Counting**: Detect and count 5 fruit types: durian, apple, dragon fruit, strawberry, coconut.

- **YOLO11n Object Detection**: Detect nearly 80 object classes including:  
  `person`, `bicycle`, `car`, `motorcycle`, `airplane`, `bus`, `train`, `truck`, `boat`, `traffic light`, `fire hydrant`, `stop sign`, `parking meter`, `bench`, `bird`, `cat`, `dog`, `horse`, `sheep`, `cow`, `elephant`, `bear`, `zebra`, `giraffe`, `backpack`, `umbrella`, `handbag`, `tie`, `suitcase`, `frisbee`, `skis`, `snowboard`, `sports ball`, `kite`, `baseball bat`, `baseball glove`, `skateboard`, `surfboard`, `tennis racket`, `bottle`, `wine glass`, `cup`, `fork`, `knife`, `spoon`, `bowl`, `banana`, `apple`, `sandwich`, `orange`, `broccoli`, `carrot`, `hot dog`, `pizza`, `donut`, `cake`, `chair`, `couch`, `potted plant`, `bed`, `dining table`, `toilet`, `tv`, `laptop`, `mouse`, `remote`, `keyboard`, `cell phone`, `microwave`, `oven`, `toaster`, `sink`, `refrigerator`, `book`, `clock`, `vase`, `scissors`, `teddy bear`, `hair dryer`, `toothbrush`.

- **Rock Paper Scissors Game**: Play Rock-Paper-Scissors with the computer.

---

## Installation Guide 

### 3.1 System Requirements

- Python 3.8 – 3.12
- GPU + CUDA (optional) for TensorFlow acceleration
- Minimum 4GB RAM (8GB or more recommended for YOLO and TensorFlow)
- Webcam required for video streaming (with `streamlit-webrtc`)

### 3.2 Required Software

Install any Python-compatible IDE, such as **Visual Studio Code**. Python 3.11 is recommended.

### 3.3 Install Required Libraries

Open a terminal (with internet connection) and run the following command:

```bash
pip install streamlit opencv-python-headless numpy streamlit-webrtc av pillow joblib ultralytics keras tensorflow torch torchvision
```

## Usage

### 4.1 Download Project Files

Clone or download all project files from the repository.

> **Note**: Due to GitHub file size limits, the `material_recognition_model.h5` file is not included.  
> You must download it manually from the link in `down.txt` and place it in the `Computer_Vision_Applications` folder.

GitHub Repository: [Dong-Quan/Computer_Vision_Applications](https://github.com/Dong-quan/Computer_Vision_Applications)

---

### 4.2 Run the App

1. Open the project folder in **VS Code**.  
2. Open a new terminal and run:

```bash
streamlit run Arun.py
```
3. Use the sample images in the `Test` folder to try out the application.

### 4.3 Expected Output

![Demo](images/demo.png)
---

## Troubleshooting

The web app may load slowly due to large models.  
If it doesn't load properly, please check the following:

- Python or library version mismatches
- Files may not be opened or placed correctly
- Webcam not working

---

## Author

- **Huynh Vo Phuc Loc**
- GitHub: [Dong Quan](https://github.com/Dong-quan)
