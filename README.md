# Basic Computer Vision guides

A computer vision toolkit for object detection, segmentation, and interactive annotation using YOLO and SAM2 models.

## 📋 Overview

ARAS provides utilities for:
- **Interactive bounding box annotation** with GUI
- **SAM2-based camera prediction** for real-time segmentation
- **YOLO result visualization** for detection and segmentation tasks
- **Streamlit deployment** for web-based inference

## 🗂️ Project Structure

```
ARAS/
├── utils/
│   ├── cam_predictor.py      # SAM2 camera predictor integration
│   ├── draw_gui.py            # Interactive box annotation GUI
│   ├── show_res.py            # YOLO results visualization
│   ├── streamlit.py           # Streamlit web app wrapper
│   ├── models/                # Model weights directory
│   └── samples/               # Sample images/data
├── data_yml/                  # Dataset configuration files
├── runs/                      # Training/inference outputs
└── setup.ipynb                # Setup and configuration notebook
```

## 🚀 Features

### 1. Interactive Bounding Box Annotation (`draw_gui.py`)

Draw bounding boxes on images with an interactive matplotlib GUI.

**Usage:**
```python
from utils.draw_gui import draw_boxes_gui

# Interactive mode
boxes = draw_boxes_gui("path/to/image.jpg")
# Returns: [(x1, y1, x2, y2), ...]
```

**Command-line:**
```bash
python utils/draw_gui.py path/to/image.jpg
```

**Controls:**
- Click and drag to draw boxes
- Press `u` to undo last box
- Press `Enter` to finish

### 2. SAM2 Camera Predictor (`cam_predictor.py`)

Real-time segmentation using SAM2 (Segment Anything Model 2) with camera input.

**Features:**
- Processes point, label, and box annotations
- Supports multi-object tracking
- Generates binary masks from predictions

**Key Functions:**
```python
# Process annotations
points, labels, boxes = process_anns(points, labels, boxes)

# Add prompts to predictor
predictor.add_new_prompt(
    frame_idx=0,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
    bbox=boxes
)
```

### 3. YOLO Results Visualization (`show_res.py`)

Visualize YOLO detection and segmentation results with ground truth comparison.

**Usage:**
```python
from utils.show_res import review_results

# Visualize results
review_results(
    result,
    print_res=True,        # Print detection details
    draw_boxes=True,       # Draw bounding boxes
    labels="path/to/labels.txt",  # Optional ground truth
    mask_alpha=0.5,        # Mask transparency
    linewidth=2            # Box line width
)
```

**Features:**
- Displays detection boxes and segmentation masks
- Overlays ground truth labels (green dashed boxes)
- Color-coded instance segmentation
- Confidence scores and class labels

**Utility Functions:**
```python
from utils.show_res import bgr2rgb

# Convert BGR to RGB
img_rgb = bgr2rgb(img)
```

### 4. Streamlit Web App (`streamlit.py`)

Deploy YOLO models as interactive web applications.

**Usage:**
```bash
# Default model (YOLOv8s-worldv2)
streamlit run utils/streamlit.py

# Custom model
streamlit run utils/streamlit.py path/to/model.pt
```

**Features:**
- Web-based inference interface
- Supports all Ultralytics models (YOLO11, YOLOv10, YOLOv8, etc.)
- Real-time predictions

## 📦 Dependencies

### Core Requirements
```
ultralytics          # YOLO models
opencv-python        # Image processing
matplotlib           # Visualization
numpy                # Numerical operations
streamlit            # Web deployment
sam2                 # Segment Anything Model 2
```

### Installation
```bash
pip install ultralytics opencv-python matplotlib numpy streamlit
```

For SAM2, follow the [official installation guide](https://github.com/facebookresearch/segment-anything-2).

## 🎯 Quick Start

### 1. Annotate Images
```python
from utils.draw_gui import draw_boxes_gui

boxes = draw_boxes_gui("image.jpg")
print(f"Annotated {len(boxes)} boxes")
```

### 2. Run YOLO Inference
```python
from ultralytics import YOLO
from utils.show_res import review_results

model = YOLO("yolov8n.pt")
results = model("image.jpg")

# Visualize
review_results(results[0])
```

### 3. Deploy with Streamlit
```bash
streamlit run utils/streamlit.py models/yolov8s-worldv2.pt
```

## 📊 Data Format

### YOLO Label Format
```
class_id x_center y_center width height
```
All coordinates are normalized (0-1).

### Bounding Box Format
```python
# draw_gui.py output
boxes = [(x1, y1, x2, y2), ...]  # Pixel coordinates
```

## 🔧 Configuration

Dataset configurations are stored in `data_yml/` directory. Example structure:
```yaml
path: /path/to/dataset
train: images/train
val: images/val
names:
  0: class1
  1: class2
```

## 📝 Notes

- **cam_predictor.py** requires SAM2 to be properly installed and configured
- Model weights should be placed in `utils/models/` directory
- Sample images are available in `utils/samples/`
- Training/inference outputs are saved to `runs/`

## 📄 License

This project is for educational purposes as part of degree project.

---

**Project:** ARAS (Advanced Recognition and Segmentation)  
**Course:** Year 3 Project  
**Last Updated:** November 2025
