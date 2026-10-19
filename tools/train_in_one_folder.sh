#!/usr/bin/env bash
# Train on the host (for example the Apple GPU through MPS) while keeping everything in ONE folder:
# virtual environment, pip and PyTorch caches, Ultralytics settings, dataset split and runs.
# Delete that folder (default ./train-workspace) and nothing is left behind.
#   EPOCHS=100 IMGSZ=640 DEVICE=mps NAME=mac-run tools/train_in_one_folder.sh
# Optional: INIT=path/to/best.pt to fine-tune, RESUME=1 to continue an interrupted run,
# BATCH=4 for large IMGSZ, HYP='{"lr0":0.002,"cos_lr":true}' for extra Ultralytics arguments,
# DEGRADE=0.4 to randomly blur / add noise / JPEG-compress 40% of training images (robustness).
set -euo pipefail
cd "$(dirname "$0")/.."

: "${EPOCHS:=100}" "${IMGSZ:=640}" "${DEVICE:=mps}" "${NAME:=mac-run}"
WORK="${WORK:-$PWD/train-workspace}"
mkdir -p "$WORK"
export PIP_CACHE_DIR="$WORK/pip-cache" YOLO_CONFIG_DIR="$WORK/ultralytics" TORCH_HOME="$WORK/torch"
export MPLCONFIGDIR="$WORK/matplotlib" PYTORCH_ENABLE_MPS_FALLBACK=1

if [ ! -x "$WORK/venv/bin/python" ]; then
  python3 -m venv "$WORK/venv"
  "$WORK/venv/bin/pip" install -r requirements-experiment.txt
fi
PY="$WORK/venv/bin/python"

if [ ! -f "$WORK/artifacts/performance-study/manifest.json" ]; then
  "$PY" prepare_grouped_split.py --output-dir "$WORK/artifacts/grouped"
  "$PY" performance_study.py prepare --grouped "$WORK/artifacts/grouped" --output "$WORK/artifacts/performance-study"
fi

args=(train --output "$WORK/artifacts/performance-study" --run "$WORK/runs" --variant improved
      --epochs "$EPOCHS" --imgsz "$IMGSZ" --device "$DEVICE" --name "$NAME")
[ -n "${BATCH:-}" ] && args+=(--batch "$BATCH")
[ -n "${HYP:-}" ] && args+=(--hyp "$HYP")
[ -n "${DEGRADE:-}" ] && args+=(--degrade "$DEGRADE")
[ -n "${INIT:-}" ] && args+=(--init "$INIT")
[ -n "${RESUME:-}" ] && args+=(--resume)
"$PY" performance_study.py "${args[@]}"

echo "Done. Weights: $WORK/runs/$NAME/weights/best.pt  Metrics: $WORK/runs/$NAME-selection.json"
echo "Copy best.pt somewhere safe, then 'rm -rf $WORK' to remove everything this script created."
