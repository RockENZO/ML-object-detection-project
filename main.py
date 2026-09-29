"""Compatibility entry point for the annotated-video CLI."""

import sys

from yolo_inference import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["predict"]))
