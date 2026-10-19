"""Validation metrics under image corruptions, to show how fast a detector degrades.

    python tools/robustness_check.py --weights best.pt --imgsz 960 --device cpu \
        --data-dir artifacts/performance-study --output reports/robustness.json

Each condition re-writes the validation images with one corruption (blur, darker,
brighter, heavy JPEG, low resolution, sensor noise), keeps the labels, and runs the
same validation as training. Validation images only: the frozen test split is never
read. The numbers say how stable the detector is on this dataset, not how it performs
on other leagues, cameras or lighting.
"""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

CONDITIONS = {
    "clean": lambda image, rng: image,
    "blur_sigma_1.5": lambda image, rng: cv2.GaussianBlur(image, (0, 0), 1.5),
    "blur_sigma_3": lambda image, rng: cv2.GaussianBlur(image, (0, 0), 3.0),
    "darker_0.6": lambda image, rng: cv2.convertScaleAbs(image, alpha=0.6),
    "brighter_1.4": lambda image, rng: cv2.convertScaleAbs(image, alpha=1.4),
    "jpeg_quality_20": lambda image, rng: cv2.imdecode(
        cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 20])[1], cv2.IMREAD_COLOR
    ),
    "half_resolution": lambda image, rng: cv2.resize(
        cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    ),
    "noise_sigma_10": lambda image, rng: np.clip(
        image.astype(np.float32) + rng.normal(0, 10, image.shape), 0, 255
    ).astype(np.uint8),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def perturbed_dataset(source, destination, condition, seed=0):
    """Write the validation images with one corruption next to unchanged labels."""
    rng = np.random.default_rng(seed)
    images = destination / "val" / "images"
    labels = destination / "val" / "labels"
    images.mkdir(parents=True)
    shutil.copytree(source / "val" / "labels", labels)
    for path in sorted((source / "val" / "images").iterdir()):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError("Cannot read " + str(path))
        cv2.imwrite(str(images / (path.stem + ".png")), CONDITIONS[condition](image, rng))
    config = destination / "data.yaml"
    config.write_text(
        'path: "%s"\ntrain: val/images\nval: val/images\nnc: 4\n'
        'names: ["ball", "goalkeeper", "player", "referee"]\n' % destination
    )
    return config


def summarise(metrics, names):
    box = metrics.box
    per_class = {
        names[int(index)]: {
            "precision": round(float(box.p[i]), 3),
            "recall": round(float(box.r[i]), 3),
            "map50": round(float(box.ap50[i]), 3),
        }
        for i, index in enumerate(box.ap_class_index)
    }
    return {
        "map50": round(float(box.map50), 3),
        "map50_95": round(float(box.map), 3),
        "ball": per_class.get("ball"),
        "classes": per_class,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/performance-study"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", nargs="*", default=list(CONDITIONS), choices=list(CONDITIONS))
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    results = {}
    with tempfile.TemporaryDirectory() as scratch:
        for condition in args.conditions:
            folder = Path(scratch) / condition
            config = perturbed_dataset(args.data_dir.resolve(), folder, condition)
            metrics = model.val(
                data=str(config), split="val", imgsz=args.imgsz, device=args.device,
                batch=4, plots=False, verbose=False, project=scratch, name="run-" + condition,
            )
            results[condition] = summarise(metrics, model.names)
            print(condition, results[condition]["map50"], results[condition]["ball"], flush=True)
    clean = results.get("clean")
    report = {
        "weights": args.weights.name,
        "checkpoint_sha256": sha256(args.weights),
        "split_manifest_sha256": sha256(args.data_dir / "manifest.json"),
        "imgsz": args.imgsz,
        "split": "validation (test split untouched)",
        "conditions": results,
    }
    if clean:
        report["map50_drop_vs_clean"] = {
            name: round(clean["map50"] - value["map50"], 3) for name, value in results.items()
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
