# Football Match Object Detection

## Description
This project aims to detect football players in videos using YOLOv5 for training and YOLOv8 for inference. The dataset is sourced from Roboflow.

## Dataset
The dataset used for this project is named "football-players-detection" and was exported via roboflow.com on December 5, 2022. It includes 663 images annotated in YOLO v5 PyTorch format. The following pre-processing and augmentation were applied to each image:
- 50% probability of horizontal flip
- Random brightness adjustment of between -20 and +20 percent

You can find the dataset [here](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc).

## Installation
To get started, install the required packages:
```python
!pip install ultralytics
!pip install roboflow
```
## Usage
Downloading the dataset
```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
version = project.version(1)
dataset = version.download("yolov5")
``` 

## Training the Model
```python
!yolo task=detect mode=train model=yolov5s.pt data={dataset.location}/data.yaml epochs=100 imgsz=640
```
## Running Inference
```python
from ultralytics import YOLO

model = YOLO('yolov8s')
results = model.predict('input_videos/08fd33_4.mp4', save=True)

print(results[0])
print('==================================================')
for box in results[0].boxes:
    print(box)
```

## License
This dataset is provided by a Roboflow user under the CC BY 4.0 license.