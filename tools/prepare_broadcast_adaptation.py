"""Create development-only full frames and ball crops; keep validation source separate."""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from match_analysis.hashing import digest

NAMES = ["ball", "goalkeeper", "player", "referee"]


def prepare(sources, partition, previous, output):
    if output.exists():
        raise ValueError("Preserve existing adaptation dataset")
    frozen = json.loads(partition.read_text())
    records = json.loads(sources.read_text())
    output.mkdir(parents=True)
    train = output / "train"
    valid = output / "val"
    for split in (train, valid):
        for kind in ("images", "labels"):
            (split / kind).mkdir(parents=True)
    provenance = []
    if previous:
        for image in sorted((previous / "images").glob("*.jpg")):
            shutil.copy2(image, train / "images" / ("legacy_" + image.name))
            shutil.copy2(
                previous / "labels" / (image.stem + ".txt"),
                train / "labels" / ("legacy_" + image.stem + ".txt"),
            )
    for item in records:
        clip = item["clip_id"]
        source = next(r for r in frozen["sources"] if r["clip_id"] == clip)
        if source["partition"] == "evaluation":
            raise ValueError("Never train or select models on final evaluation sources")
        labels = Path(item["labels"])
        data = json.loads(labels.read_text())
        info = data["info"]
        gt = Path(source["annotations"])
        if (
            source["annotations_sha256"] != digest(gt)
            or source["match_id"] != "SoccerNet-GSR-game-" + info["game_id"]
        ):
            raise ValueError("Frozen annotation provenance changed")
        frames = json.loads(gt.read_text())["frames"]
        by_image = {}
        for a in data["annotations"]:
            by_image.setdefault(a["image_id"], []).append(a)
        image_info = {int(Path(f["file_name"]).stem) - 1: f for f in data["images"]}
        target = train if source["partition"] == "development" else valid
        # Validation frames are sampled more sparsely; they never generate training crops.
        for f in frames[:: 1 if source["partition"] == "development" else 3]:
            meta = image_info[f["frame_index"]]
            image = labels.parent / "img1" / meta["file_name"]
            frame = Image.open(image).convert("RGB")
            width, height = frame.size
            boxes = []
            for a in by_image.get(meta["image_id"], []):
                role = a.get("attributes", {}).get("role")
                b = a.get("bbox_image")
                if role not in NAMES or not b:
                    continue
                x1, y1 = max(0, b["x"]), max(0, b["y"])
                x2, y2 = min(width, b["x"] + b["w"]), min(height, b["y"] + b["h"])
                if x2 > x1 and y2 > y1:
                    boxes.append([NAMES.index(role), x1, y1, x2, y2])

            def save(name, img, bounds):
                left, top, right, bottom = bounds
                cw, ch = right - left, bottom - top
                lines = []
                for c, x1, y1, x2, y2 in boxes:
                    a, b = max(x1, left), max(y1, top)
                    d, e = min(x2, right), min(y2, bottom)
                    if (
                        d <= a
                        or e <= b
                        or (d - a) * (e - b) / ((x2 - x1) * (y2 - y1)) < 0.7
                    ):
                        continue
                    lines.append(
                        f"{c} {(a + d - 2 * left) / (2 * cw):.8f} {(b + e - 2 * top) / (2 * ch):.8f} {(d - a) / cw:.8f} {(e - b) / ch:.8f}"
                    )
                img.save(target / "images" / (name + ".jpg"), quality=95)
                (target / "labels" / (name + ".txt")).write_text(
                    "\n".join(lines) + "\n"
                )

            stem = f"{clip}_{f['frame_index']:06d}"
            save(stem, frame, (0, 0, width, height))
            if source["partition"] == "development":
                for ball in [b for b in boxes if b[0] == 0][:1]:
                    _, x1, y1, x2, y2 = ball
                    size = min(320, width, height)
                    rng = random.Random(f["frame_index"])
                    left = max(
                        0,
                        min(
                            width - size,
                            int(
                                (x1 + x2) / 2
                                - size / 2
                                + rng.uniform(-0.25, 0.25) * size
                            ),
                        ),
                    )
                    top = max(
                        0,
                        min(
                            height - size,
                            int(
                                (y1 + y2) / 2
                                - size / 2
                                + rng.uniform(-0.25, 0.25) * size
                            ),
                        ),
                    )
                    save(
                        stem + "_ballcrop",
                        frame.crop((left, top, left + size, top + size)),
                        (left, top, left + size, top + size),
                    )
            frame.close()
        provenance.append(
            {
                "clip_id": clip,
                "match_id": source["match_id"],
                "partition": source["partition"],
                "annotation_sha256": source["annotations_sha256"],
                "source_label_sha256": digest(labels),
            }
        )
    dataset = output / "dataset.yaml"
    dataset.write_text(
        "path: "
        + json.dumps(str(output.resolve()))
        + "\ntrain: train/images\nval: val/images\nnames: "
        + json.dumps(NAMES)
        + "\n"
    )
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "frozen_partition_sha256": digest(partition),
                "sources": provenance,
                "previous_dataset": str(previous) if previous else None,
                "policy": "Development full frames plus native 320-pixel ball crops. Validation is a different source match; no evaluation data included.",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "Training images",
        len(list((train / "images").glob("*"))),
        "validation",
        len(list((valid / "images").glob("*"))),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for field in ("sources", "partition", "output"):
        p.add_argument("--" + field, required=True, type=Path)
    p.add_argument("--previous", type=Path)
    a = p.parse_args()
    prepare(a.sources, a.partition, a.previous, a.output)
