# Training the detector

The detector is a four-class YOLOv8s (ball, goalkeeper, player, referee). `performance_study.py` builds the grouped split and trains it; the container and the one-folder script below only wrap that command so nothing is installed or left behind on your machine.

## Pick where to train

| Where | Command | Speed | Cleanup |
| --- | --- | --- | --- |
| Mac, Apple GPU (MPS) | `tools/train_in_one_folder.sh` | Fast: the only option that uses the Apple GPU | Delete `train-workspace/` |
| Docker on a Mac or any CPU | `docker compose -f docker-compose.train.yml run --rm train` | Slow (roughly 5 min per epoch on 4 cores at 640 px) | `docker compose -f docker-compose.train.yml down --volumes --rmi all` |
| Docker on a Linux or cloud machine with an NVIDIA GPU | see below | Fast | same as above |

Docker Desktop on macOS runs Linux inside a virtual machine and cannot reach the Apple GPU, so a container on a Mac trains on CPU only. For the Apple GPU use the one-folder script: it puts the virtual environment, pip and PyTorch caches, Ultralytics settings, dataset split and runs in a single directory (`./train-workspace`, or `WORK=/some/path`), so deleting that directory removes everything.

## Settings

Both routes read the same environment variables.

| Variable | Default | Meaning |
| --- | --- | --- |
| `EPOCHS` | 100 | Epoch budget; early stopping (patience 20) ends a run sooner when validation stops improving |
| `IMGSZ` | 640 | Training image size; try 960 or 1280 to help the ball, which is a few pixels wide (slower, more memory) |
| `DEVICE` | `mps` for the script, `cpu` for Docker | `0` for the first NVIDIA GPU |
| `NAME` | `mac-run` / `container-run` | Run name; use a new one per experiment |
| `INIT` | none | Checkpoint to fine-tune from instead of the COCO weights |
| `RESUME` | none | Any value continues an interrupted run called `NAME` from its own `last.pt` |

The split is rebuilt from the dataset in `training/` with a fixed seed and is identical to the one behind the published numbers (same groups, same 342 / 122 / 135 train, validation and test images). Only validation is used for selection; the test split stays untouched.

## Examples

```bash
# Mac with the Apple GPU: fine-tune the 72-epoch detector at a larger size to help the ball
mkdir -p train-input && cp best-72epoch-stripped.pt train-input/best.pt   # from the project files
EPOCHS=40 IMGSZ=960 INIT=train-input/best.pt NAME=ball-960 tools/train_in_one_folder.sh

# Docker, CPU: fresh 100-epoch run at 640 px
docker compose -f docker-compose.train.yml run --rm train

# Docker, NVIDIA GPU, fine-tune at 960 px (the input folder is mounted at /input)
mkdir -p train-input train-output && cp best-72epoch-stripped.pt train-input/best.pt
TORCH_INDEX=https://download.pytorch.org/whl/cu128 DEVICE=0 EPOCHS=40 IMGSZ=960 \
  INIT=/input/best.pt NAME=ball-960 \
  docker compose -f docker-compose.train.yml --profile gpu run --rm --build train-gpu
```

Fine-tuning from a finished run is allowed here because that run trained on the same training split only. The report records the starting checkpoint's hash.

## Results and cleanup

- Container: `train-output/<NAME>/` receives `best.pt` (weights without optimiser state, about 22 MB), `results.csv` and `selection.json` (validation metrics, options, hashes). Copy it somewhere safe, then run the `down --volumes --rmi all` command above.
- One-folder script: weights are in `train-workspace/runs/<NAME>/weights/best.pt` and metrics in `train-workspace/runs/<NAME>-selection.json`. Copy them out, then delete `train-workspace/`.

Selection reports are validation-only. A final test evaluation is a separate, deliberate step (`finalize_study.py`, which reads the canonical `base` and `improved` run names only).

## Checking robustness

`tools/robustness_check.py` re-runs the validation split with one corruption at a time (blur, darker, brighter, heavy JPEG, half resolution, sensor noise) and writes the metrics per condition:

```bash
python tools/robustness_check.py --weights best.pt --imgsz 1280 --output reports/robustness.json
```

It reads validation images only. It shows how stable a detector is on this dataset, not how it generalises to other leagues, cameras or lighting; that needs an independent test set. Reports for the detectors fine-tuned so far are in `reports/robustness_*_20261019.json`.

## Weights

Trained weights are not committed to `main`. Finished detectors are stored as a single `best.pt` plus a README (with its sha256 and validation metrics) on branches named `weights-*`, for example `weights-ball-1280-robust` (the detector behind the committed demo). Fetch one without switching branches:

```bash
git fetch origin weights-ball-1280-robust && git show origin/weights-ball-1280-robust:best.pt > train-input/best.pt
```

## Licence of what you train

Weights trained with Ultralytics are covered by AGPL-3.0, like the rest of this repository (see [LICENSE](../LICENSE) and [NOTICE.md](../NOTICE.md)).
