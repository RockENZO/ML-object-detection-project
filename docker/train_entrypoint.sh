#!/usr/bin/env bash
# Runs inside the training container. Settings come from environment variables:
#   EPOCHS (100)  IMGSZ (640)  DEVICE (cpu, or 0 for the first NVIDIA GPU)  NAME (container-run)
#   INIT  optional checkpoint to fine-tune from, e.g. /input/best.pt
#   RESUME  any non-empty value continues an interrupted run called NAME
# The split and training images are read from the mounted ./training directory.
set -euo pipefail
cd /app

: "${EPOCHS:=100}" "${IMGSZ:=640}" "${DEVICE:=cpu}" "${NAME:=container-run}"

if [ ! -f artifacts/performance-study/manifest.json ]; then
  python prepare_grouped_split.py
  python performance_study.py prepare
fi

args=(train --variant improved --epochs "$EPOCHS" --imgsz "$IMGSZ" --device "$DEVICE" --name "$NAME")
[ -n "${INIT:-}" ] && args+=(--init "$INIT")
[ -n "${RESUME:-}" ] && args+=(--resume)
python performance_study.py "${args[@]}"

# Hand the result to the host: weights without optimiser state, plus the metrics.
out="/output/$NAME"
mkdir -p "$out"
python - "$out" "runs/performance-study/$NAME" <<'PY'
import shutil, sys
from pathlib import Path
from ultralytics.utils.torch_utils import strip_optimizer
out, run = Path(sys.argv[1]), Path(sys.argv[2])
strip_optimizer(str(run / "weights/best.pt"), str(out / "best.pt"))
shutil.copy(run / "results.csv", out / "results.csv")
shutil.copy(run.parent / (run.name + "-selection.json"), out / "selection.json")
PY
echo "Done. Results are in ./train-output/$NAME on the host."
