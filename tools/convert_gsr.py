"""Convert official GSR 1.3 labels to evaluation samples; never infer transitions."""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from match_analysis.hashing import digest


def convert(labels, video, output):
    data = json.loads(Path(labels).read_text())
    info = data["info"]
    if float(info["version"]) < 1.3:
        raise ValueError("GSR annotation version >=1.3 required")
    by_image = {}
    for row in data["annotations"]:
        by_image.setdefault(row["image_id"], []).append(row)
    frames = []
    next_due = 0
    fps = float(info["frame_rate"])
    for image in sorted(data["images"], key=lambda x: x["file_name"]):
        index = int(Path(image["file_name"]).stem) - 1
        t = index / fps
        if t < next_due - 1e-8:
            continue
        while next_due <= t + 1e-8:
            next_due += 0.1
        people = []
        ball = None
        for row in by_image.get(image["image_id"], []):
            box = row.get("bbox_image")
            role = row.get("attributes", {}).get("role")
            if not box or role not in ("player", "goalkeeper", "referee", "ball"):
                continue
            xy = None
            pitch = row.get("bbox_pitch")
            if pitch:
                point = [pitch["x_bottom_middle"] + 52.5, pitch["y_bottom_middle"] + 34]
                if (
                    all(math.isfinite(v) for v in point)
                    and 0 <= point[0] <= 105
                    and 0 <= point[1] <= 68
                ):
                    xy = point
            record = {
                "id": str(row["track_id"]),
                "role": role,
                "team": row["attributes"].get("team"),
                "bbox": [box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"]],
                "xy": xy,
            }
            if role == "ball":
                ball = record
            else:
                people.append(record)
        frames.append(
            {"frame_index": index, "timestamp": t, "people": people, "ball": ball}
        )
    result = {
        "schema_version": 1,
        "clip_id": info["name"],
        "match_id": "SoccerNet-GSR-game-" + info["game_id"],
        "source_annotation_sha256": digest(labels),
        "video_sha256": digest(video),
        "annotation_version": info["version"],
        "team_label_permutation_invariant": True,
        "pitch_dimensions": [105, 68],
        "frames": frames,
        "notes": [
            "GSR native coordinates centred at midfield; shifted +52.5,+34.",
            "Outside-pitch positions omitted from spatial errors, retained for detection matching.",
            "No human eligible/replay/possession-change annotations supplied.",
            "Anonymous colour clusters are evaluated up to global A/B label permutation; this does not validate supplied team names.",
        ],
    }
    output = Path(output)
    if output.exists():
        raise ValueError("Do not overwrite existing ground truth")
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("labels", "video", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    result = convert(args.labels, args.video, args.output)
    print("Converted", len(result["frames"]), "sampled frames")
