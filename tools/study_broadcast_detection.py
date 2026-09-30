"""Cache real development detections once; measure precision/recall and tracker profiles."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2

from match_analysis.detection import predict
from match_analysis.hashing import digest


def main():
    p = argparse.ArgumentParser()
    for field in ("video", "annotations", "checkpoint", "output"):
        p.add_argument("--" + field, required=True, type=Path)
    p.add_argument("--device", default="mps")
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--tile-size", type=int, default=640)
    p.add_argument("--tiled-classes", default="0")
    p.add_argument("--partition", required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Preserve prior study output")
    os.environ["YOLO_CONFIG_DIR"] = str(args.output.parent / "ultralytics-config")
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    from ultralytics import YOLO

    model = YOLO(args.checkpoint)
    gt = json.loads(args.annotations.read_text())
    truth_document = json.loads(args.annotations.read_text())
    frozen = json.loads(args.partition.read_text())
    source = next(
        (r for r in frozen["sources"] if r["clip_id"] == truth_document["clip_id"]),
        None,
    )
    if not source or source["partition"] != "development":
        raise ValueError("Tuning studies require frozen development sources")
    if source["video_sha256"] != digest(args.video) or source[
        "annotations_sha256"
    ] != digest(args.annotations):
        raise ValueError("Frozen study source changed")
    cap = cv2.VideoCapture(str(args.video))
    records = []
    start = time.perf_counter()
    try:
        for truth in gt["frames"][:: args.stride]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, truth["frame_index"])
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Cannot decode annotation frame")
            result = predict(
                model,
                frame,
                args.device,
                confidence=0.03,
                tile_size=args.tile_size,
                tiled_classes=tuple(map(int, args.tiled_classes.split(","))),
            )
            records.append(
                {
                    "frame_index": truth["frame_index"],
                    "timestamp": truth["timestamp"],
                    "boxes": result.boxes.data.cpu().numpy().tolist(),
                }
            )
            if len(records) % 20 == 0:
                print("Measured", len(records), "frames", flush=True)
    finally:
        cap.release()
    args.output.write_text(
        json.dumps(
            {
                "partition": "development",
                "clip_id": gt["clip_id"],
                "match_id": gt["match_id"],
                "video_sha256": digest(args.video),
                "annotation_sha256": digest(args.annotations),
                "model_sha256": digest(args.checkpoint),
                "tile_size": args.tile_size,
                "tiled_classes": args.tiled_classes,
                "stride": args.stride,
                "compute_seconds": time.perf_counter() - start,
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
