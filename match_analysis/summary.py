"""Statistics exclude predictions and expose denominators and missing coverage."""

import csv
import json
import math
from collections import Counter, deque
from pathlib import Path

import numpy as np


def load_frames(output):
    for path in sorted((Path(output) / "shots").glob("*.jsonl")):
        with path.open() as file:
            for line in file:
                yield json.loads(line)


def direction_at(metadata, t, team):
    for interval in metadata.get("attacking_directions", []):
        if interval["start"] <= t < interval["end"]:
            return interval[team]
    return None


def validate_metadata(metadata):
    previous = -1
    for interval in metadata.get("attacking_directions", []):
        if set(interval) != {"start", "end", "A", "B"} or not all(
            isinstance(interval[k], (int, float)) and math.isfinite(interval[k])
            for k in interval
        ):
            raise ValueError("Direction intervals need finite start,end,A,B")
        if (
            interval["start"] < previous
            or interval["end"] <= interval["start"]
            or interval["A"] not in (-1, 1)
            or interval["B"] != -interval["A"]
        ):
            raise ValueError(
                "Direction intervals must not overlap and teams must attack opposite directions"
            )
        previous = interval["end"]
    return metadata


def summarize(output, config, metadata, video_end):
    output = Path(output)
    frames = iter(load_frames(output))
    current = next(frames, None)
    frame_count = 0
    heatmaps = {team: np.zeros((14, 21), dtype=int) for team in ("A", "B")}
    player_zones = {team: Counter() for team in ("A", "B")}
    track_file = (output / "tracks.csv").open("w", newline="")
    track_writer = csv.DictWriter(
        track_file,
        fieldnames=[
            "timestamp",
            "shot",
            "track_id",
            "role",
            "team",
            "observation",
            "x",
            "y",
        ],
    )
    track_writer.writeheader()
    durations = Counter()
    zones = {"A": Counter(), "B": Counter()}
    events = []
    trends = []
    window = deque()
    observation_counts = Counter()
    while current is not None:
        frame = current
        current = next(frames, None)
        frame_count += 1
        t = frame["timestamp"]
        next_time = (
            current["timestamp"]
            if current is not None
            else min(video_end, t + 1 / config.sample_hz)
        )
        dt = max(0, min(next_time - t, 2 / config.sample_hz))
        durations["sampled_seconds"] += dt
        if frame["eligible"]:
            durations["eligible_seconds"] += dt
        team = frame["possession"]
        covered = frame["eligible"] and team in ("A", "B")
        if covered:
            durations[team] += dt
        entry = {"t": t, "dt": dt, "team": team if covered else None, "pressure": False}
        if covered:
            direction = direction_at(metadata, t, team)
            x = frame["ball"]["xy"][0]
            entry["pressure"] = direction is not None and (
                x >= config.pitch_length / 2
                if direction == 1
                else x < config.pitch_length / 2
            )
            entry["direction_known"] = direction is not None
            zones[team][min(2, int(x / config.pitch_length * 3))] += dt
        window.append(entry)
        while window and window[0]["t"] < t - config.pressure_window:
            window.popleft()
        total = sum(r["dt"] for r in window if r["team"])
        known = sum(r["dt"] for r in window if r.get("direction_known"))
        trends.append(
            {
                "timestamp": t,
                "eligible": frame["eligible"],
                "possession": team,
                "covered_seconds": total,
                "unknown_seconds": sum(r["dt"] for r in window if not r["team"]),
                "A_possession_share": sum(r["dt"] for r in window if r["team"] == "A")
                / total
                if total
                else None,
                "A_territorial_pressure": sum(
                    r["dt"] for r in window if r["team"] == "A" and r["pressure"]
                )
                / total
                if total and known >= total - 1e-8
                else None,
                "B_territorial_pressure": sum(
                    r["dt"] for r in window if r["team"] == "B" and r["pressure"]
                )
                / total
                if total and known >= total - 1e-8
                else None,
            }
        )
        if frame.get("event"):
            events.append(frame["event"])
        for person in frame["people"]:
            observation_counts[person["observation"]] += 1
            track_writer.writerow(
                {
                    "timestamp": t,
                    "shot": frame["shot"],
                    "track_id": person["id"],
                    "role": person["role"],
                    "team": person["team"],
                    "observation": person["observation"],
                    "x": person["xy"][0] if person["xy"] else None,
                    "y": person["xy"][1] if person["xy"] else None,
                }
            )
            if (
                frame["eligible"]
                and person["observation"] == "observed"
                and person["role"] == "player"
                and person.get("team") in heatmaps
                and person.get("xy") is not None
            ):
                x, y = person["xy"]
                column = min(20, int(x / config.pitch_length * 21))
                row = min(13, int(y / config.pitch_width * 14))
                heatmaps[person["team"]][row, column] += 1
                player_zones[person["team"]][
                    min(2, int(x / config.pitch_length * 3))
                ] += dt
    track_file.close()
    np.savez_compressed(output / "heatmaps.npz", **heatmaps)
    covered = durations["A"] + durations["B"]
    eligible = durations["eligible_seconds"]
    result = {
        "schema_version": 1,
        "status": "experimental",
        "sampled_frames": frame_count,
        "durations": dict(durations),
        "unknown_eligible_seconds": max(0, eligible - covered),
        "excluded_seconds": max(0, durations["sampled_seconds"] - eligible),
        "possession_coverage": covered / eligible if eligible else 0,
        "eligible_fraction": eligible / durations["sampled_seconds"]
        if durations["sampled_seconds"]
        else 0,
        "observed_possession_percent": {
            team: 100 * durations[team] / covered if covered else None
            for team in ("A", "B")
        },
        "zone_occupancy_seconds": {team: dict(value) for team, value in zones.items()},
        "player_zone_observation_seconds": {
            team: dict(value) for team, value in player_zones.items()
        },
        "possession_changes": len(events),
        "track_observation_counts": dict(observation_counts),
        "team_names": metadata.get("team_names", {"A": "Team A", "B": "Team B"}),
        "limitations": [
            "Wide/replay segmentation and airborne filtering are heuristics pending independent validation.",
            "Possession is proximity-based estimated control; possession changes are not official tackles or turnovers.",
            "IDs are shot-local; offscreen players and predicted points are not statistical observations.",
            "Pitch dimensions are assumed unless supplied; no full-match official statistics are inferred.",
        ],
    }
    for name, value in [("summary", result), ("events", events), ("trends", trends)]:
        (output / (name + ".json")).write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n"
        )
    with (output / "events.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "type",
                "method",
                "observed_support_seconds",
                "support_track_ids",
                "shot",
                "timestamp",
                "from_team",
                "to_team",
                "confirmed_at",
                "uncertainty",
            ],
        )
        writer.writeheader()
        writer.writerows(events)
    return result
