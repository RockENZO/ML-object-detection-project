"""Adapt the existing four-class checkpoint using frozen development/validation sources."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from match_analysis.hashing import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--device", default="mps")
    p.add_argument("--epochs", type=int, default=8)
    a = p.parse_args()
    if a.run.exists():
        raise ValueError("Preserve previous training evidence")
    provenance = json.loads((a.dataset.parent / "provenance.json").read_text())
    if any(
        r["partition"] not in ("development", "validation")
        for r in provenance["sources"]
    ):
        raise ValueError("Final evaluation data cannot be used in training")
    os.environ["YOLO_CONFIG_DIR"] = str(a.run.parent / "ultralytics-config")
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import torch
    import ultralytics
    from ultralytics import YOLO

    model = YOLO(a.checkpoint)
    if list(model.names.values()) != ["ball", "goalkeeper", "player", "referee"]:
        raise ValueError(
            "Adaptation requires the trained four-class football checkpoint"
        )
    dataset_files = {
        str(p.relative_to(a.dataset.parent)): digest(p)
        for split in ("train", "val")
        for p in sorted((a.dataset.parent / split).rglob("*"))
        if p.is_file()
    }
    if not dataset_files:
        raise ValueError("Adaptation dataset is empty")
    started = time.perf_counter()
    model.train(
        data=str(a.dataset),
        epochs=a.epochs,
        batch=4,
        imgsz=640,
        device=a.device,
        workers=0,
        seed=20260930,
        deterministic=True,
        freeze=10,
        lr0=0.001,
        cos_lr=True,
        mosaic=0.2,
        close_mosaic=2,
        optimizer="AdamW",
        amp=False,
        patience=a.epochs,
        cache=False,
        plots=False,
        project=str(a.run.parent),
        name=a.run.name,
        exist_ok=False,
        save_period=1,
    )
    result = {
        "initial_checkpoint_sha256": digest(a.checkpoint),
        "dataset_provenance_sha256": digest(a.dataset.parent / "provenance.json"),
        "data_sources": provenance["sources"],
        "best_checkpoint_sha256": digest(a.run / "weights/best.pt"),
        "dataset_files": dataset_files,
        "epochs": a.epochs,
        "imgsz": 640,
        "freeze_backbone_layers": 10,
        "device": a.device,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "compute_seconds": time.perf_counter() - started,
        "selection": "Best validation aggregate detection fitness; final evaluation match never used.",
    }
    (a.run / "adaptation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
