# ML Object Detection Project

## Project Title and Description
**Detect football players in videos using YOLOv5 for training and YOLOv8 for inference**

This project focuses on detecting football players in video footage using state-of-the-art object detection models. The approach leverages YOLOv5 for the training phase and YOLOv8 for inference, combining the strengths of both versions for optimal performance.

## Dataset Information
The dataset used for this project is sourced from Roboflow and contains:
- **663 annotated images** specifically labeled for football player detection
- Format: YOLO v5 PyTorch format
- Pre-processing applied:
  - 50% probability of horizontal flip
  - Random brightness adjustment between -20% and +20%

The dataset is named "football-players-detection" and was exported from Roboflow on December 5, 2022. You can access the original dataset [here](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc).

## Prerequisites
Before running this project, ensure you have the following installed:
- Python 3.x (recommended 3.8 or higher)
- pip (Python package manager)
- Required Python packages:
  - ultralytics
  - opencv-python
  - roboflow

## Installation Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/RockENZO/ML-object-detection-project.git
   cd ML-object-detection-project
   ```

2. Install the required dependencies:
   ```bash
   pip install ultralytics opencv-python roboflow
   ```

## Usage Guide

### Training the Model
The training process uses YOLOv5 on the Roboflow dataset. The dataset has already been downloaded and organized in the `training/` directory.

To train the model:
```bash
# Navigate to the project directory
cd ML-object-detection-project

# Train using the provided data.yaml configuration
yolo task=detect mode=train model=yolov5s.pt data=training/football-players-detection-1/data.yaml epochs=100 imgsz=640
```

Training parameters can be adjusted as needed:
- `epochs`: Number of training epochs (default: 100)
- `imgsz`: Image size for training (default: 640)
- `model`: YOLOv5 variant to use (yolov5s.pt, yolov5m.pt, yolov5l.pt, etc.)

### Running Inference
For inference, the project uses YOLOv8 via the `yolo_inference.py` script:

```bash
# Run the inference script
python yolo_inference.py
```

This will:
1. Load the YOLOv8s model
2. Process the sample video (`input_videos/08fd33_4.mp4`)
3. Save the results with bounding boxes to the `runs/detect/predict/` directory
4. Print detection results to the console

To run inference on a different video, modify the `yolo_inference.py` script:
```python
# Change this line to point to your video
results = model.predict('path/to/your/video.mp4', save=True)
```

## Project Structure
```
ML-object-detection-project/
├── input_videos/                 # Input video files for processing
│   └── 08fd33_4.mp4             # Sample input video
├── models/                      # Pre-trained model weights
│   ├── yolov5su.pt              # YOLOv5 small model
│   └── yolov8s.pt               # YOLOv8 small model
├── runs/                        # Output from training and inference
│   └── detect/                  # Detection results
│       ├── predict/             # Latest inference results
│       ├── predict1/            # Previous inference runs
│       └── predict2/            # Earlier inference runs
├── training/                    # Training dataset and configuration
│   └── football-players-detection-1/
│       ├── data.yaml           # Dataset configuration for YOLO
│       ├── football-players-detection-1/  # Actual dataset
│       │   ├── train/          # Training images and labels
│       │   ├── val/            # Validation images and labels
│       │   └── test/           # Test images and labels
│       ├── README.dataset.txt  # Dataset information
│       └── README.roboflow.txt # Roboflow export details
├── tracker/                     # Object tracking utilities
│   ├── __init__.py
│   └── tracker.py               # Tracking implementation
├── utils/                       # Utility functions
│   ├── __init__.py
│   └── video_utils.py           # Video processing helpers
├── yolo_inference.py            # Main inference script (YOLOv8)
├── main.py                      # Alternative entry point
└── README.md                    # This file
```

## Model Details
- **Training Model**: YOLOv5 (selected for its proven performance and stability in training scenarios)
- **Inference Model**: YOLOv8s (chosen for its improved speed and accuracy in real-time applications)

## License
This dataset is provided by a Roboflow user under the CC BY 4.0 license. Please refer to the original Roboflow dataset page for specific licensing details.

## Acknowledgments
- Roboflow for providing the annotated dataset
- Ultralytics for the YOLOv5 and YOLOv8 implementations
- The open-source computer vision community