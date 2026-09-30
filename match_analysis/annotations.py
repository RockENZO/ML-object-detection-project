"""Validate independently reviewed labels; never create ground truth from predictions."""

import copy
import math


def reviewed(truth, intervals, transitions, reviewer, source_hash):
    if not reviewer.strip():
        raise ValueError("Human reviewer required")
    result = copy.deepcopy(truth)
    intervals = sorted(intervals, key=lambda x: x["start"])
    prior = 0
    for interval in intervals:
        start, end = interval["start"], interval["end"]
        if (
            not all(
                isinstance(t, (int, float)) and math.isfinite(t) for t in (start, end)
            )
            or start < prior
            or end <= start
        ):
            raise ValueError("Intervals must be finite, positive and non-overlapping")
        if interval["view"] not in ("eligible", "replay", "ineligible", "uncertain"):
            raise ValueError("Unknown view label")
        prior = end
    for frame in result["frames"]:
        for field in ("eligible", "replay"):
            frame.pop(field, None)
        matches = [v for v in intervals if v["start"] <= frame["timestamp"] < v["end"]]
        if not matches:
            raise ValueError(
                "Every sampled frame needs a reviewed interval; use uncertain explicitly"
            )
        if matches[0]["view"] != "uncertain":
            frame["eligible"] = matches[0]["view"] == "eligible"
            frame["replay"] = matches[0]["view"] == "replay"
    for event in transitions:
        if (
            not math.isfinite(event["timestamp"])
            or event["timestamp"] < 0
            or event["from_team"] == event["to_team"]
        ):
            raise ValueError("Invalid reviewed transition")
    result["transitions"] = sorted(transitions, key=lambda x: x["timestamp"])
    result["transitions_use_ground_truth_team_labels"] = True
    result["review"] = {
        "reviewer": reviewer.strip(),
        "source_annotation_sha256": source_hash,
        "intervals": intervals,
        "complete_video_review_confirmed": True,
    }
    return result
