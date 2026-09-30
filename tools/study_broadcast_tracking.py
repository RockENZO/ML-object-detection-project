"""Development-only tracker profile comparison on cached 10 Hz detections."""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from match_analysis.evaluation import iou
from match_analysis.hashing import digest


def score(rows, truth):
    pairs = Counter()
    ng = npred = 0
    for actual, predicted in zip(truth, rows):
        gs = actual["people"]
        ng += len(gs)
        npred += len(predicted)
        for a in gs:
            for b in predicted:
                if iou(a["bbox"], b["bbox"]) >= 0.5:
                    pairs[(a["id"], b["id"])] += 1
    ids = sorted({x for x, y in pairs})
    tracks = sorted({y for x, y in pairs})
    tp = 0
    if ids and tracks:
        matrix = np.array([[pairs[x, y] for y in tracks] for x in ids])
        r, c = linear_sum_assignment(-matrix)
        tp = int(matrix[r, c].sum())
    return {
        "idf1": 2 * tp / (ng + npred) if ng + npred else None,
        "IDTP": tp,
        "IDFP": npred - tp,
        "IDFN": ng - tp,
    }


def main():
    p = argparse.ArgumentParser()
    for field in ("video", "detections", "annotations", "output"):
        p.add_argument("--" + field, required=True, type=Path)
    p.add_argument("--partition", required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Do not overwrite measured evidence")
    os.environ["YOLO_CONFIG_DIR"] = str(args.output.parent / "ultralytics-config")
    import ultralytics
    import yaml
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.bot_sort import BOTSORT

    cache = json.loads(args.detections.read_text())
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
    if cache["video_sha256"] != digest(args.video) or cache[
        "annotation_sha256"
    ] != digest(args.annotations):
        raise ValueError("Detection cache does not belong to these frozen sources")
    gt = json.loads(args.annotations.read_text())["frames"]
    truth = {f["frame_index"]: f for f in gt}
    base = yaml.safe_load(
        (Path(ultralytics.__file__).parent / "cfg/trackers/botsort.yaml").read_text()
    )
    results = []
    profiles = [
        ("baseline", 0.25, 0.25, 30, True),
        ("long_buffer", 0.25, 0.25, 90, True),
        ("clean_tracks", 0.4, 0.4, 90, True),
        ("clean_no_fuse", 0.4, 0.4, 90, False),
        ("low_match", 0.15, 0.25, 90, False),
    ]
    trackers = []
    for name, high, new, buffer, fuse in profiles:
        settings = {
            **base,
            "track_high_thresh": high,
            "new_track_thresh": new,
            "track_buffer": buffer,
            "fuse_score": fuse,
        }
        trackers.append(
            {
                "name": name,
                "settings": settings,
                "tracker": BOTSORT(SimpleNamespace(**settings), frame_rate=10),
                "rows": [],
            }
        )
    cap = cv2.VideoCapture(str(args.video))
    try:
        for cached in cache["records"]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cached["frame_index"])
            ok, image = cap.read()
            if not ok:
                raise RuntimeError("Cannot decode source frame")
            boxes = np.array(
                [b for b in cached["boxes"] if b[5] != 0 and b[4] >= 0.1],
                dtype=np.float32,
            ).reshape(-1, 6)
            for profile in trackers:
                tracks = profile["tracker"].update(Boxes(boxes, image.shape[:2]), image)
                profile["rows"].append(
                    [{"id": str(int(t[4])), "bbox": t[:4].tolist()} for t in tracks]
                )
    finally:
        cap.release()
    for profile in trackers:
        result = {
            "profile": profile["name"],
            "settings": {
                k: profile["settings"][k]
                for k in (
                    "track_high_thresh",
                    "new_track_thresh",
                    "track_buffer",
                    "fuse_score",
                )
            },
            "metrics": score(
                profile["rows"], [truth[f["frame_index"]] for f in cache["records"]]
            ),
        }
        results.append(result)
        print(result, flush=True)
    args.output.write_text(
        json.dumps(
            {
                "partition": "development",
                "detector_cache_sha256": digest(args.detections),
                "annotation_sha256": digest(args.annotations),
                "profiles": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
