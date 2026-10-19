# Licence and third-party notices

This repository is licensed under the **GNU Affero General Public License v3.0** (see [LICENSE](LICENSE)). The code imports Ultralytics (AGPL-3.0), so distributing it, or trained weights, under a more permissive licence would conflict with Ultralytics' terms. Trained weights published from this project are released under the same AGPL-3.0 terms.

| Component | Licence | How it is used |
| --- | --- | --- |
| [Ultralytics](https://github.com/ultralytics/ultralytics) YOLOv8 and its pretrained weights | AGPL-3.0 | Imported by the training and analysis code; `models/yolov8s.pt` and `models/yolov5su.pt` are Ultralytics weights |
| [PnLCalib](https://github.com/mguti97/PnLCalib) | GPL-2.0 | Downloaded separately by `tools/setup_pitch.py` for pitch calibration; its code is not copied into this repository |
| [football-players-detection](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) dataset (Roboflow Universe) | CC BY 4.0 | Training images and labels in `training/`; attribution to the dataset's authors is required |
| Other Python dependencies (OpenCV, NumPy, SciPy, Shapely, PyTorch and others) | Their own permissive or BSD-style licences | Installed from PyPI; see each package |

## Demo footage

`input_videos/08fd33_4.mp4` and the clips derived from it in `web/data/` are broadcast footage whose rights holder is not stated in this repository. They are included only to demonstrate the viewer. Replace them with openly licensed or self-owned footage before any commercial or wide public use, and remove them on a rights holder's request.
