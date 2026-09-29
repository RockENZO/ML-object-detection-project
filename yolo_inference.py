"""Run YOLO inference or evaluation on the bundled football dataset."""

import argparse
import json
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
    metrics = model.val(data=str(args.data), split="test", imgsz=args.imgsz,
                        plots=True, **run_options)
    summary = {
        "model": str(args.model.resolve()),
        "dataset": str(args.data.resolve()),
        "split": "test",
        "imgsz": args.imgsz,
        "box_precision": float(metrics.box.mp),
        "box_recall": float(metrics.box.mr),
        "box_map50": float(metrics.box.map50),
        "box_map50_95": float(metrics.box.map),
    }
    report = args.output_dir / "metrics.json"
    report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Metrics saved to {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
