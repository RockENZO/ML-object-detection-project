#!/usr/bin/env bash
# Runs inside the training container. Settings come from environment variables:
#   EPOCHS (100)  IMGSZ (640)  DEVICE (cpu, or 0 for the first NVIDIA GPU)  NAME (container-run)
#   INIT  optional checkpoint to fine-tune from, e.g. /input/best.pt
#   RESUME  any non-empty value continues an interrupted run called NAME
# The split and training images are read from the mounted ./training directory.
set -euo pipefail
cd /app

: "${EPOCHS:=100}" "${IMGSZ:=640}" "${DEVICE:=cpu}" "${NAME:=container-run}"
case "$NAME" in
  .|..|*/*|*\\*) echo "NAME must be a single directory name" >&2; exit 2 ;;
esac
# Fail before loading data/training if the host export mount is not writable.
out="/output/$NAME"
if ! mkdir -p "$out" || ! probe=$(mktemp "$out/.write-check.XXXXXX"); then
  echo "Export directory must be writable by container UID $(id -u): $out" >&2
  exit 1
fi
rm "$probe"

if [ ! -f artifacts/performance-study/manifest.json ]; then
  python prepare_grouped_split.py
  python performance_study.py prepare
fi

args=(train --variant improved --epochs "$EPOCHS" --imgsz "$IMGSZ" --device "$DEVICE" --name "$NAME")
[ -n "${BATCH:-}" ] && args+=(--batch "$BATCH")
[ -n "${HYP:-}" ] && args+=(--hyp "$HYP")
[ -n "${DEGRADE:-}" ] && args+=(--degrade "$DEGRADE")
[ -n "${INIT:-}" ] && args+=(--init "$INIT")
[ -n "${RESUME:-}" ] && args+=(--resume)
python performance_study.py "${args[@]}"

# Hand the result to the host, with hashes of both source and stripped bytes.
python export_training.py "$out" "runs/performance-study/$NAME"
echo "Done. Results are in ./train-output/$NAME on the host."
