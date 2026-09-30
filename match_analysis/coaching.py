"""Evidence-bounded geometry for coaching review; never infer a full formation."""

import csv
import json
from pathlib import Path

import numpy as np

from .summary import load_frames


def frame_features(frame, config, minimum=6):
    result = {
        "timestamp": frame["timestamp"],
        "shot": frame["shot"],
        "eligible": frame["eligible"],
        "ball_status": "airborne_uncertain"
        if frame["ball"].get("airborne_uncertain")
        else frame["ball"].get("reason", frame["ball"]["observation"]),
        "ball_observation": frame["ball"]["observation"],
        "possession": frame["possession"],
        "teams": {},
    }
    for team in ("A", "B"):
        players = [
            p
            for p in frame["people"]
            if p["role"] == "player"
            and p.get("team") == team
            and p["observation"] == "observed"
            and p.get("xy") is not None
            and p.get("confidence", 0) >= 0.5
        ]
        # Duplicate identities cannot inflate count or geometric support.
        players = list({p["id"]: p for p in players}.values())
        n = len(players)
        reason = (
            "excluded_interval"
            if not frame["eligible"]
            else "insufficient_observed_players"
            if n < minimum
            else "implausible_outfield_count"
            if n > 10
            else "observed_visible_group"
        )
        features = {
            "observed_outfield_count": n,
            "reason": reason,
            "scope": "visible_group_only",
            "centroid": None,
            "width_m": None,
            "depth_m": None,
            "spread_m": None,
        }
        if reason == "observed_visible_group":
            xy = np.asarray([p["xy"] for p in players])
            centroid = xy.mean(axis=0)
            features.update(
                centroid=centroid.tolist(),
                width_m=float(np.ptp(xy[:, 1])),
                depth_m=float(np.ptp(xy[:, 0])),
                spread_m=float(np.sqrt(np.mean(np.sum((xy - centroid) ** 2, axis=1)))),
            )
        result["teams"][team] = features
    return result


def export(output, config):
    output = Path(output)
    records = []
    next_due = 0
    for frame in load_frames(output):
        if frame["timestamp"] + 1e-8 < next_due:
            continue
        while next_due <= frame["timestamp"] + 1e-8:
            next_due += 1
        records.append(frame_features(frame, config))
    report = {
        "schema_version": 1,
        "status": "experimental",
        "scope": "Observed visible groups. No formation, defensive line, running load or whole-team tactical inference.",
        "minimum_observed_outfield_players": 6,
        "records": records,
    }
    (output / "coaching.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    with (output / "coaching.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "timestamp",
                "shot",
                "team",
                "observed_outfield_count",
                "reason",
                "scope",
                "width_m",
                "depth_m",
                "spread_m",
            ],
        )
        writer.writeheader()
        for row in records:
            for team, feature in row["teams"].items():
                writer.writerow(
                    {
                        "timestamp": row["timestamp"],
                        "shot": row["shot"],
                        "team": team,
                        **{k: v for k, v in feature.items() if k != "centroid"},
                    }
                )
    return report
