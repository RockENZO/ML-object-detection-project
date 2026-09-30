# Football player detection

This repository contains a four-class Roboflow football dataset (ball, goalkeeper, player, referee), sample video, pretrained YOLO weights, and scripts for annotated inference and evaluation.

## Setup

Use Python 3.10 or newer. From the repository root:

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
python prepare_grouped_split.py
python run_experiment.py --epochs 10 --imgsz 320 --device cpu
```

Use the resulting `best.pt` checkpoint for a single, held-out **test** evaluation:

```bash
python yolo_inference.py --model runs/grouped-baseline/train/weights/best.pt evaluate --data artifacts/grouped/data.yaml
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

## Grouped split and checkpoint validation

The original export has 612 train, 38 validation and 13 test images. Filename-derived video prefixes overlap between original splits, so random frame splits may overstate generalization. `prepare_grouped_split.py` validates annotations and creates a deterministic seed-42 split with entire filename-prefix groups assigned together: 477 train, 122 validation and 64 test images. It stores per-image/label hashes, group assignment and source paths in `artifacts/grouped/manifest.json`. Original inputs are retained. Filename prefixes are provisional video identities; upstream confirmation and match/near-duplicate auditing remain necessary.

Use Python 3.12 and `pip install -r requirements-experiment.txt` for the reviewed experiment runtime. `run_experiment.py` fits a fresh four-class model using training and validation groups, saves a training/checkpoint manifest, and evaluates the untouched test groups after training. The default ten-epoch, 320-pixel run is a short transfer-learning baseline, not proof of convergence or production quality. Increase training duration and select parameters using validation data only, then use a new frozen test study for additional performance claims.

Evaluation refuses the bundled 80-class COCO checkpoint and any class order differing from ball/goalkeeper/player/referee. It verifies every test image/label against the split manifest, and reports test groups, checkpoint/split hashes, runtime version, box precision/recall, mAP50 and mAP50-95. YOLO box precision is computed by the pinned Ultralytics evaluator; it is not an image-level correctness rate or a manually measured workflow saving.

## Measured baseline (2026-09-30)

Ten epochs at 320 pixels, batch 8, seed 42, MPS device, initialized from the bundled YOLOv8s checkpoint. The selected checkpoint was chosen using validation groups before test evaluation.

| Test metric | Result |
| --- | ---: |
| Images / provisional video groups | 64 / 3 |
| Mean box precision | 0.35248 |
| Mean box recall | 0.29783 |
| mAP50 | 0.29878 |
| mAP50-95 | 0.14464 |

[Per-class test report](reports/test_baseline_20260930.json), [training parameters and checkpoint hashes](reports/training_baseline_20260930.json), [split audit](reports/grouped_split_audit.json) and [validation learning curve](reports/training_curve_20260930.csv) are included. The trained checkpoint remains a local generated artifact, not a bundled downloadable model; reproduce it using the script. MPS reports some nondeterministic operations, so a fixed seed does not promise byte-identical weights across runs/devices.

These results do not support 97.5% precision or production readiness. Main next steps are validation-driven longer/higher-resolution training, error analysis for small balls and visually similar player roles, confirmed match/video identities and a separately frozen test study. Do not tune parameters on the published test results.
