# Football player detection

This repository contains a four-class Roboflow football dataset (ball, goalkeeper, player, referee), sample video, pretrained YOLO weights, and scripts for annotated inference and evaluation.

## Setup

Use Python 3.9 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The dataset is from [Roboflow Universe](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1). Its export identifies the dataset license as CC BY 4.0. Check the source terms before redistributing the images.

## Annotated inference

```bash
python yolo_inference.py predict
python yolo_inference.py predict --input input_videos/08fd33_4.mp4 --output-dir runs/my-prediction
```

The default model is the bundled `models/yolov8s.pt`. This is a **pretrained** checkpoint; this repository does not include a documented fine-tuned checkpoint. Pass `--model path/to/your/best.pt` to evaluate or use a separately trained model. Predictions with drawn boxes are saved under the requested output directory. `python main.py predict` provides the same interface.

## Training and evaluation

The checked-in dataset has train, validation, and test image and label folders. Training uses the training split and validation uses the validation split:

```bash
yolo detect train model=models/yolov8s.pt data=training/football-players-detection-1/data.yaml epochs=100 imgsz=640
```

Use the resulting `best.pt` checkpoint for a single, held-out **test** evaluation:

```bash
python yolo_inference.py --model runs/detect/train/weights/best.pt evaluate
```

The command writes `runs/codex-evaluate/metrics.json` with box precision, recall, mAP@0.5, and mAP@0.5:0.95. Record the checkpoint, dataset version, image size, Ultralytics version, and exact command with any performance claim. The test split must remain untouched during model selection. Do not compare a YOLOv5 training result with inference from different YOLOv8 weights as though they were one model.

The profile's **97.5% precision** and **85% less manual review** figures cannot currently be reproduced from the checked-in code: there is no saved test report, evaluation command, or manual-review study. Treat them as unverified until their underlying results are published.

## Project layout

- `yolo_inference.py`: annotated prediction and test evaluation CLI
- `main.py`: compatibility entry point to the same CLI
- `models/`: bundled pretrained weights
- `training/football-players-detection-1/`: Roboflow dataset and `data.yaml`
- `input_videos/`: sample input
- `runs/`: generated predictions and evaluation results
