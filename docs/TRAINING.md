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
| `BATCH` | 8 | Batch size; use 4 at `IMGSZ=1280` (about 10 GB on an Apple GPU) |
| `HYP` | none | JSON of extra Ultralytics train arguments, e.g. `'{"lr0":0.002,"cos_lr":true,"close_mosaic":8}'`; recorded in the selection report |
| `DEGRADE` | none | Probability (e.g. `0.4`) of randomly blurring, adding noise or JPEG-compressing each training image, so the detector copes with soft or noisy broadcast frames |
| `RESUME` | none | Any value continues an interrupted run called `NAME` from its own `last.pt` and original `<NAME>-training.json` sidecar |

The split is rebuilt from the dataset in `training/` with a fixed seed and is identical to the one behind the published numbers (same groups, same 342 / 122 / 135 train, validation and test images). Only validation is used for selection; the test split stays untouched.

## Examples

Fetch the existing published 72-epoch checkpoint first; it is not a file bundled
on `main`. This only downloads weights, without switching the source branch:

```bash
mkdir -p train-input train-output
git fetch origin weights-72ep
git show FETCH_HEAD:best.pt > train-input/best.pt
```

```bash
# Mac with the Apple GPU: fine-tune the fetched detector at a larger size
EPOCHS=40 IMGSZ=960 INIT=train-input/best.pt NAME=ball-960 tools/train_in_one_folder.sh

# Docker, CPU: fresh 100-epoch run at 640 px
mkdir -p train-input train-output
docker compose -f docker-compose.train.yml run --rm train

# Docker, NVIDIA GPU, fine-tune at 960 px (the input folder is mounted at /input)
mkdir -p train-input train-output  # best.pt was fetched above
TORCH_INDEX=https://download.pytorch.org/whl/cu128 DEVICE=0 EPOCHS=40 IMGSZ=960 \
  INIT=/input/best.pt NAME=ball-960 \
  docker compose -f docker-compose.train.yml --profile gpu run --rm --build train-gpu
```

The container runs as UID 1000. Create `train-output` on the host before launch
and ensure that UID can write there. Compose refuses to create a missing export
mount as root, and the entrypoint checks write access **before training**. A
permission mismatch fails early; these tools do not change ownership of your
existing files automatically.

`BATCH`, `HYP` and `DEGRADE` are forwarded through both Docker services as well as
the one-folder script. `HYP` must be an object of learning/augmentation settings;
runner keys such as `name`, `project`, `data`, `resume`, `epochs`, `imgsz`, `batch`
and `device` must use their dedicated options. `DEGRADE` must be finite and in
`[0, 1]`, so 40% is `0.4`, not `40`.

### Resuming without changing the experiment

Each new run writes `<NAME>-training.json` beside its run directory before the
first training call. Keep that sidecar and the full run directory (`last.pt`,
including optimizer state). Resume restores the original initialization hash,
split/crop hashes and degradation probability even if `INIT`, `DEGRADE` and
`HYP` are omitted. Hyperparameters in the report come from the saved checkpoint;
explicit conflicting `HYP`, `INIT` or `DEGRADE` values are rejected. If a new
run failed before writing its first `last.pt`, `RESUME=1` safely restarts it from
the recorded initialization path and original options (the original weights must
still exist and match their hash). This starts at epoch zero, not from a
checkpoint; the report marks `restarted_before_first_checkpoint`. A run with
`best.pt` but missing `last.pt` is not restarted over its existing weights.

Runs created before this metadata existed cannot be safely inferred from
`last.pt` alone: the external degradation hook was not stored there. Such runs
are preserved but this runner refuses to resume them without the original
sidecar. Use a new run name rather than inventing provenance or deleting the old
run. The degradation RNG restarts from its recorded seed on resume, so this is
not a promise of a bit-identical uninterrupted run.

The one-folder installer records a success marker only after the pinned
requirements install completes; rerunning after an interrupted installation
retries it. Changed requirements also trigger installation again.

Fine-tuning from a finished run is allowed here because that run trained on the same training split only. The report records the starting checkpoint's hash.

## Results and cleanup

- Container: `train-output/<NAME>/` receives `best.pt` (weights without optimiser state, about 22 MB), `results.csv` and `selection.json` (validation metrics, options, hashes) and `training.json` (original run settings). Copy it somewhere safe, then run the `down --volumes --rmi all` command above.
- One-folder script: weights are in `train-workspace/runs/<NAME>/weights/best.pt` and metrics in `train-workspace/runs/<NAME>-selection.json`. Copy them out, then delete `train-workspace/`.

In an exported report, `checkpoint_sha256` identifies the **stripped exported**
`best.pt`; `training_checkpoint_sha256` identifies the original checkpoint before
stripping. The configuration sidecar must also match its recorded hash before
any export artifacts are written. The training run's own selection report remains unchanged.

Selection reports are validation-only. A final test evaluation is a separate, deliberate step (`finalize_study.py`, which reads the canonical `base` and `improved` run names only).

## Recipe for the published detectors

The detectors behind the committed demo were produced on an Apple GPU by chaining three fine-tunes, each starting from the previous `best.pt` and selected on the validation split only:

```bash
# Assumes the fetch/setup above and the default train-workspace directory.
# 1. 72-epoch detector -> 960 px
EPOCHS=40 IMGSZ=960 INIT=train-input/best.pt NAME=ball-960 tools/train_in_one_folder.sh
# 2. 960 px -> 1280 px, cosine schedule, stronger augmentation
EPOCHS=30 IMGSZ=1280 BATCH=4 INIT=train-workspace/runs/ball-960/weights/best.pt NAME=ball-1280 \
  HYP='{"lr0":0.002,"lrf":0.05,"cos_lr":true,"warmup_epochs":1,"close_mosaic":8,"scale":0.6,"hsv_v":0.5,"hsv_s":0.6,"degrees":3.0,"translate":0.15,"mixup":0.1,"erasing":0.3}' \
  tools/train_in_one_folder.sh
# 3. 1280 px with blur, motion blur, noise and JPEG degradation on 40% of training images
EPOCHS=25 IMGSZ=1280 BATCH=4 DEGRADE=0.4 INIT=train-workspace/runs/ball-1280/weights/best.pt NAME=ball-1280-robust \
  HYP='{"lr0":0.001,"lrf":0.05,"cos_lr":true,"warmup_epochs":1,"close_mosaic":5,"scale":0.6,"hsv_v":0.5,"hsv_s":0.6,"degrees":3.0,"translate":0.15,"mixup":0.1,"erasing":0.3}' \
  tools/train_in_one_folder.sh
```

Each step took roughly 45 to 60 minutes on one Apple GPU (about 70 s per epoch at 960 px, 125 s at 1280 px). The degradations in step 3 overlap the corruptions in `tools/robustness_check.py`, so the robustness numbers for that detector are a stress test of known degradations, not evidence about unseen conditions.

## Checking robustness

`tools/robustness_check.py` re-runs the validation split with one corruption at a time (blur, darker, brighter, heavy JPEG, half resolution, sensor noise) and writes the metrics per condition:

```bash
python tools/robustness_check.py --weights best.pt --imgsz 1280 --output reports/robustness.json
```

It reads validation images only. It shows how stable a detector is on this dataset, not how it generalises to other leagues, cameras or lighting; that needs an independent test set. Reports for the detectors fine-tuned so far are in `reports/robustness_*_20261019.json`.

## Weights

Trained weights are not committed to `main`. Finished detectors are stored as a single `best.pt` plus a README (with its sha256 and validation metrics) on branches named `weights-*`, for example `weights-ball-1280-robust` (the detector behind the committed demo). Fetch one without switching branches:

```bash
mkdir -p train-input
git fetch origin weights-ball-1280-robust
git show FETCH_HEAD:best.pt > train-input/best.pt
```

## Licence of what you train

Weights trained with Ultralytics are covered by AGPL-3.0, like the rest of this repository (see [LICENSE](../LICENSE) and [NOTICE.md](../NOTICE.md)).
