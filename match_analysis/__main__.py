"""Portable commands: python -m match_analysis analyze/evaluate/freeze."""

import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Experimental offline broadcast football analysis"
    )
    subs = parser.add_subparsers(dest="command", required=True)
    analysis = subs.add_parser("analyze")
    for name in ("video", "checkpoint", "output"):
        analysis.add_argument("--" + name, required=True, type=Path)
    for name in ("config", "metadata", "pitch-root"):
        analysis.add_argument("--" + name, type=Path)
    analysis.add_argument("--device", default="cpu", help="cpu, mps or CUDA device 0")
    analysis.add_argument("--resume", action="store_true")
    analysis.add_argument("--max-seconds", type=float)
    evaluation = subs.add_parser("evaluate")
    for name in ("run", "annotations", "output"):
        evaluation.add_argument("--" + name, required=True, type=Path)
    evaluation.add_argument("--partition", required=True, type=Path)
    freeze = subs.add_parser("freeze")
    freeze.add_argument("--sources", required=True, type=Path)
    freeze.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "analyze":
            os.environ.setdefault(
                "YOLO_CONFIG_DIR",
                str(args.output.resolve().parent / "ultralytics-config"),
            )
            Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("YOLO_AUTOINSTALL", "false")
            from .config import Config
            from .pipeline import analyze

            result = analyze(
                args.video,
                args.checkpoint,
                args.output,
                Config.load(args.config),
                json.loads(args.metadata.read_text()) if args.metadata else None,
                args.pitch_root,
                args.device,
                args.resume,
                args.max_seconds,
            )
        elif args.command == "freeze":
            from .evaluation import freeze

            result = freeze(args.sources, args.output)
        else:
            from .evaluation import evaluate

            result = evaluate(args.run, args.annotations, args.partition, args.output)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.exit(2, str(error) + "\n")


if __name__ == "__main__":
    main()
