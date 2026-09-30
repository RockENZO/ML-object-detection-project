"""Run YOLO inference or evaluation on the bundled football dataset."""

import argparse
import json
import hashlib
import importlib.metadata
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "yolov8s.pt"
DEFAULT_DATA = ROOT / "training" / "football-players-detection-1" / "data.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="Path to model weights (default: bundled pretrained YOLOv8s)")
    subcommands = parser.add_subparsers(dest="command", required=True)

    predict = subcommands.add_parser("predict", help="Save annotated video or image predictions")
    predict.add_argument("--input", type=Path, default=ROOT / "input_videos" / "08fd33_4.mp4")
    predict.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "codex-predict")

    evaluate = subcommands.add_parser("evaluate", help="Evaluate on the held-out test split")
    evaluate.add_argument("--data", type=Path, default=DEFAULT_DATA)
    evaluate.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "codex-evaluate")
    evaluate.add_argument("--imgsz", type=int, default=640)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.model.is_file():
        raise SystemExit(f"Model weights not found: {args.model}")

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_options = {"project": str(args.output_dir.parent),
                   "name": args.output_dir.name, "exist_ok": True}

    if args.command == "predict":
        if not args.input.is_file():
            raise SystemExit(f"Input file not found: {args.input}")
        model.predict(source=str(args.input), save=True, **run_options)
        print(f"Annotated predictions saved under {args.output_dir}")
        return 0

    if not args.data.is_file():
        raise SystemExit(f"Dataset YAML not found: {args.data}")
    import yaml
    from prepare_grouped_split import NAMES, digest
    dataset = yaml.safe_load(args.data.read_text())
    names = dataset.get('names', [])
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names)]
    model_names = model.names
    if isinstance(model_names, dict):
        model_names = [model_names[key] for key in sorted(model_names)]
    if list(names) != NAMES or list(model_names) != NAMES:
        raise SystemExit('Evaluation requires a checkpoint trained for ball/goalkeeper/player/referee in that order; bundled COCO weights are incompatible')
    manifest = args.data.parent / 'manifest.json'
    if not manifest.is_file():
        raise SystemExit('Evaluation requires grouped split manifest; run prepare_grouped_split.py first')
    split = json.loads(manifest.read_text())
    root = Path(dataset['path'])
    test_records = [record for record in split['records'] if record['split'] == 'test']
    if not test_records:
        raise SystemExit('Test split is empty')
    for record in test_records:
        for kind, key, hash_key in [('images', 'image', 'image_sha256'), ('labels', 'label', 'label_sha256')]:
            artifact = root / 'test' / kind / Path(record[key]).name
            if not artifact.is_file() or digest(artifact) != record[hash_key]:
                raise SystemExit('Test artifact differs from split manifest: ' + str(artifact))
    metrics = model.val(data=str(args.data), split="test", imgsz=args.imgsz,
                        plots=True, **run_options)
    summary = {
        "model": args.model.name,
        "model_sha256": digest(args.model),
        "split_manifest_sha256": digest(manifest),
        "test_images": len(test_records),
        "test_groups": sorted({record['group'] for record in test_records}),
        "ultralytics_version": importlib.metadata.version('ultralytics'),
        "dataset": "grouped/data.yaml",
        "split": "test",
        "imgsz": args.imgsz,
        "per_class": metrics.summary(),
        "box_precision": float(metrics.box.mp),
        "box_recall": float(metrics.box.mr),
        "box_map50": float(metrics.box.map50),
        "box_map50_95": float(metrics.box.map),
    }
    report = args.output_dir / "metrics.json"
    report.write_text(json.dumps(summary, indent=2, default=lambda value: value.item()) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=lambda value: value.item()))
    print(f"Metrics saved to {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
