---
title: AutoNav AI - Off-Road Autonomous Navigation
emoji: 🚙
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.30.0
app_file: app.py
pinned: false
license: mit
---

# AutoNav AI: Off-Road Traversability and Autonomous Obstacle Avoidance

> **Smart India Hackathon (SIH26126) / 5th-Semester Major Project**  
> Real-time deep learning vision pipeline combining **SegFormer-B0** semantic terrain segmentation and **YOLOv8** obstacle detection with an interactive 3D WebGL car simulation.

---

## 🌟 Key Features
- **Terrain Traversability Analysis:** Fine-tuned SegFormer-B0 classifies unstructured off-road paths (trail, grass, mud, obstacle, sky).
- **Real-Time Obstacle Detection:** YOLOv8 Nano detects dynamic hazards (vehicles, pedestrians, rocks, animals).
- **Costmap Fusion & Directional Planner:** 3-sector spatial clearance algorithm calculating optimal steering angles and obstacle avoidance.
- **Interactive 3D Simulation:** WebGL Three.js simulation with 19-beam LiDAR, dynamic patrol obstacles, mud slip physics, and auto-reverse recovery.
- **Real-Time Performance:** Runs at ~43 FPS (SegFormer) and ~87 FPS (YOLO) with ~45 ms end-to-end latency on RTX 2050 GPU (also CPU compatible).

---

## 🚀 Live Demo & Hosting Options

### 1. Hugging Face Spaces (Cloud Demo)
This repository is pre-configured for instant deployment on **Hugging Face Spaces**:
1. Create a new Space on [huggingface.co/new-space](https://huggingface.co/new-space).
2. Choose **Streamlit** as the Space SDK.
3. Upload or push these files. The app will build and run automatically!

### 2. Run Locally
```bash
# Activate your environment
activate.bat

# Launch the Streamlit application
run_app.bat
```
Visit `http://localhost:8501` in your browser.

---

## 📁 Project Structure
```text
├── app.py                             # Main Streamlit web application
├── car_3d_simulation.html             # High-difficulty 3D Three.js simulation
├── AutoNav_AI_Project_Report.html     # Academic project report (printable to PDF)
├── requirements.txt                   # Cloud-ready Python dependencies
├── packages.txt                       # Linux system libraries for cloud hosting
└── autonomousPathFinder/
    ├── models/                        # Pretrained & fine-tuned SegFormer + YOLO weights
    ├── datasets/                      # Lightweight sample datasets (RUGD, RELLIS-3D, BDD100K)
    ├── src/                           # Pipeline modules (segmentation, detection, costmap, planner)
    └── outputs/                       # Fine-tuning curves & evaluation metrics
```
