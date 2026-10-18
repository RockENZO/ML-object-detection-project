"""Momentum, critical moments and tactic notes derived from saved observations.

Everything here runs on CPU from a finished analysis directory. Predicted points
never count as evidence. Where the ball is not observed the calibrated camera
centre ("camera focus") stands in for the location of play; every value that
uses it says so, and nothing here is an official statistic.
"""

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace

import numpy as np

from .summary import direction_at, load_frames

TEAMS = ("A", "B")
# Penalty areas are fixed-size under the laws of the game, whatever the pitch size.
BOX_DEPTH, BOX_WIDTH = 16.5, 40.32


@dataclass
class Settings:
    bin_seconds: float = 1.0
    min_player_confidence: float = 0.5
    ball_radius_m: float = 12.0
    focus_radius_m: float = 18.0
    nearest_players: int = 6
    min_superiority_players: int = 4
    max_anchor_distance_m: float = 25.0
    proximity_radius_m: float = 6.0
    ball_max_player_distance_m: float = 15.0
    tempo_ratio: float = 1.2
    tempo_min_bins: int = 3
    smooth_sigma_bins: float = 2.5
    min_smoothed_support: float = 0.15
    weights: dict = field(
        default_factory=lambda: {
            "control": 0.3,
            "proximity": 0.2,
            "superiority": 0.3,
            "territory": 0.2,
        }
    )
    min_weight: float = 0.3
    swing_level: float = 0.2
    swing_total: float = 0.6
    swing_gap_bins: int = 12
    pressure_level: float = 0.35
    pressure_min_bins: int = 5
    box_min_seconds: float = 1.5
    fast_ball_mps: float = 16.0
    crowd_players: int = 8
    crowd_radius_m: float = 8.0
    crowd_min_seconds: float = 1.5
    shift_metres: float = 12.0
    shift_window: float = 5.0
    min_outfield: int = 6
    min_shape_change: float = 3.0
    # Derived from the run's sample rate in build(); defaults suit 10 Hz.
    max_gap: float = 0.3
    speed_tolerance: float = 0.1
    max_moments: int = 14


def finite_xy(xy):
    return (
        xy is not None
        and len(xy) == 2
        and all(isinstance(v, (int, float)) and math.isfinite(v) for v in xy)
    )


def observed_players(frame, minimum_confidence):
    """Unique observed, assigned outfield players with pitch coordinates."""
    seen = {}
    for person in frame["people"]:
        if (
            person["role"] == "player"
            and person.get("team") in TEAMS
            and person["observation"] == "observed"
            and finite_xy(person.get("xy"))
            and person.get("confidence", 0) >= minimum_confidence
        ):
            seen[person["id"]] = person
    return list(seen.values())


def camera_focus(frame, size, config):
    """Pitch point under the image centre; a proxy for where play is."""
    matrix = (frame.get("calibration") or {}).get("homography")
    if not frame["eligible"] or matrix is None or size is None:
        return None
    value = np.asarray(matrix, dtype=float) @ np.array([size[0] / 2, size[1] / 2, 1.0])
    if abs(value[2]) < 1e-8:
        return None
    xy = value[:2] / value[2]
    if (
        not np.all(np.isfinite(xy))
        or not 0 <= xy[0] <= config.pitch_length
        or not 0 <= xy[1] <= config.pitch_width
    ):
        return None
    return [float(xy[0]), float(xy[1])]


def in_box(xy, config):
    """'left' or 'right' penalty area containing the point, else None."""
    half = BOX_WIDTH / 2
    if abs(xy[1] - config.pitch_width / 2) > half:
        return None
    if xy[0] <= BOX_DEPTH:
        return "left"
    if xy[0] >= config.pitch_length - BOX_DEPTH:
        return "right"
    return None


def other(team):
    return "B" if team == "A" else "A"


def names_of(metadata):
    return metadata.get("team_names", {"A": "Team A", "B": "Team B"})


def collect(output, config, metadata, size, settings):
    """One pass over saved frames: per-frame records and per-bin aggregates."""
    frames, bins = [], defaultdict(lambda: defaultdict(list))
    history = defaultdict(deque)
    previous_ball = None
    ball_speeds = []
    for frame in load_frames(output):
        t = frame["timestamp"]
        key = int(t // settings.bin_seconds)
        bin_ = bins[key]
        players = observed_players(frame, settings.min_player_confidence)
        ball = frame["ball"]
        raw_ball = (
            ball["xy"]
            if frame["eligible"]
            and ball["observation"] == "observed"
            and finite_xy(ball.get("xy"))
            else None
        )
        # Sanity gate: a ball in play has someone within reach. Isolated
        # detections (adverts, spots, boots) fall back to the camera focus.
        ball_xy = raw_ball
        if raw_ball is not None and not any(
            finite_xy(p.get("xy"))
            and p["observation"] == "observed"
            and math.dist(raw_ball, p["xy"]) <= settings.ball_max_player_distance_m
            for p in frame["people"]
        ):
            ball_xy = None
        focus = camera_focus(frame, size, config)
        if ball_xy is not None:
            anchor, kind, radius = ball_xy, "ball", settings.ball_radius_m
        elif focus is not None:
            anchor, kind, radius = focus, "camera_focus", settings.focus_radius_m
        else:
            anchor, kind, radius = None, None, None
        record = {
            "t": t,
            "shot": frame["shot"],
            "eligible": frame["eligible"],
            "players": players,
            "ball": ball_xy,
            "ball_raw": raw_ball,
            "focus": focus,
            "airborne": bool(ball.get("airborne_uncertain")),
            "anchor": anchor,
            "anchor_kind": kind,
            "possession": frame["possession"] if frame["eligible"] else "unknown",
            "everyone": [
                p
                for p in frame["people"]
                if p["observation"] == "observed"
                and p["role"] in ("player", "goalkeeper")
                and finite_xy(p.get("xy"))
            ],
        }
        frames.append(record)
        bin_["n"].append(1)
        if frame["eligible"]:
            bin_["eligible"].append(1)
        if record["possession"] in TEAMS:
            bin_["control"].append(record["possession"])
        if anchor is not None:
            bin_["anchor"].append(anchor)
            bin_["anchor_kind"].append(kind)
            ranked = sorted(players, key=lambda p: math.dist(anchor, p["xy"]))
            near = [
                p
                for p in ranked[: settings.nearest_players]
                if math.dist(anchor, p["xy"]) <= settings.max_anchor_distance_m
            ]
            if kind == "ball" and ranked:
                closest = ranked[0]
                if math.dist(anchor, closest["xy"]) <= settings.proximity_radius_m:
                    bin_["proximity"].append(1 if closest["team"] == "A" else -1)
            counts = {team: sum(1 for p in near if p["team"] == team) for team in TEAMS}
            if sum(counts.values()) >= settings.min_superiority_players:
                bin_["superiority"].append(
                    (counts["A"] - counts["B"]) / (counts["A"] + counts["B"])
                )
                bin_["superiority_counts"].append((counts["A"], counts["B"]))
            direction = direction_at(metadata, t, "A")
            if direction is not None:
                signed = (anchor[0] - config.pitch_length / 2) * direction
                bin_["territory"].append(signed / (config.pitch_length / 2))
        if ball_xy is not None:
            same = (
                previous_ball is not None
                and previous_ball[2] == frame["shot"]
                and 0 < t - previous_ball[0] <= settings.max_gap
            )
            if same:
                speed = math.dist(ball_xy, previous_ball[1]) / (t - previous_ball[0])
                if speed <= 40:
                    ball_speeds.append((t, speed, previous_ball[1], ball_xy))
                else:
                    ball_speeds.append(None)
            elif ball_speeds:
                # Time proximity alone cannot join observations across a cut.
                ball_speeds.append(None)
            previous_ball = (t, ball_xy, frame["shot"])
        else:
            previous_ball = None
            if ball_speeds:
                ball_speeds.append(None)
        for person in players:
            samples = history[person["id"]]
            samples.append((t, person["xy"]))
            while t - samples[0][0] > 1.0 + settings.speed_tolerance + 0.5:
                samples.popleft()
            earlier = [
                s
                for s in samples
                if abs(t - s[0] - 1.0) <= settings.speed_tolerance
            ]
            if earlier:
                speed = math.dist(person["xy"], earlier[0][1]) / (t - earlier[0][0])
                if speed <= 12:
                    bin_["speed"].append(speed)
    return frames, bins, ball_speeds


def bin_series(bins, settings, metadata):
    keys = sorted(bins)
    first, last = keys[0], keys[-1]
    base = sum(w for k, w in settings.weights.items() if k != "territory")
    rows = []
    for key in range(first, last + 1):
        bin_ = bins.get(key)
        row = {"t": key * settings.bin_seconds, "raw": None, "support": 0.0}
        rows.append(row)
        # Territory can only exist where attacking direction is known.
        possible = base + (
            settings.weights["territory"]
            if direction_at(metadata, row["t"] + settings.bin_seconds / 2, "A") is not None
            else 0.0
        )
        if not bin_:
            continue
        n, eligible = len(bin_["n"]), len(bin_["eligible"])
        row["eligible_fraction"] = eligible / n if n else 0
        parts = {}
        covered = bin_["control"]
        if len(covered) >= 3:
            a, b = covered.count("A"), covered.count("B")
            parts["control"] = (a - b) / (a + b)
        if len(bin_["proximity"]) >= 3:
            parts["proximity"] = float(np.mean(bin_["proximity"]))
        if len(bin_["superiority"]) >= 3:
            parts["superiority"] = float(np.mean(bin_["superiority"]))
        if len(bin_["territory"]) >= 3:
            parts["territory"] = float(np.clip(np.mean(bin_["territory"]), -1, 1))
        weight = sum(settings.weights[k] for k in parts)
        row["components"] = {k: round(v, 3) for k, v in parts.items()}
        if weight >= settings.min_weight:
            row["raw"] = sum(settings.weights[k] * v for k, v in parts.items()) / weight
            row["support"] = min(1.0, weight / possible)
        if bin_["anchor"]:
            row["anchor"] = np.median(np.asarray(bin_["anchor"]), axis=0).tolist()
            kinds = bin_["anchor_kind"]
            row["anchor_kind"] = "ball" if kinds.count("ball") * 2 >= len(kinds) else "camera_focus"
        if bin_["superiority_counts"]:
            counts = np.mean(np.asarray(bin_["superiority_counts"]), axis=0)
            row["near_anchor"] = [round(float(counts[0]), 1), round(float(counts[1]), 1)]
        if len(bin_["speed"]) >= 8:
            row["intensity_mps"] = round(float(np.mean(bin_["speed"])), 2)
    return rows


def smooth(rows, settings):
    """Support-weighted Gaussian smoothing; gaps stay gaps."""
    sigma = settings.smooth_sigma_bins
    radius = int(math.ceil(3 * sigma))
    result = []
    for i in range(len(rows)):
        total = weight = norm = 0.0
        for j in range(max(0, i - radius), min(len(rows), i + radius + 1)):
            kernel = math.exp(-0.5 * ((i - j) / sigma) ** 2)
            norm += kernel
            if rows[j]["raw"] is not None:
                total += kernel * rows[j]["support"] * rows[j]["raw"]
                weight += kernel * rows[j]["support"]
        support = weight / norm if norm else 0.0
        if support >= settings.min_smoothed_support and weight > 0:
            result.append({"value": total / weight, "support": min(1.0, support)})
        else:
            result.append({"value": None, "support": 0.0})
    return result


def runs(series, predicate):
    """Maximal index runs where predicate(value, support) holds."""
    start = None
    for i, item in enumerate(series + [{"value": None, "support": 0.0}]):
        ok = item["value"] is not None and predicate(item["value"], item["support"])
        if ok and start is None:
            start = i
        if not ok and start is not None:
            yield start, i - 1
            start = None


def ball_share(rows, first, last):
    """Share of bins in [first, last] whose location of play was the observed ball."""
    kinds = [rows[i].get("anchor_kind") for i in range(first, last + 1)]
    return kinds.count("ball") / len(kinds)


def confidence_word(support, share=1.0):
    """Medium needs decent support and mostly observed-ball evidence."""
    return "medium" if support >= 0.45 and share >= 0.5 else "low"


def momentum_moments(rows, series, settings, names):
    moments = []
    positive = list(runs(series, lambda v, s: v >= settings.swing_level and s >= 0.3))
    negative = list(runs(series, lambda v, s: v <= -settings.swing_level and s >= 0.3))
    labelled = sorted(
        [(a, b, "A") for a, b in positive] + [(a, b, "B") for a, b in negative]
    )
    for (a0, a1, ta), (b0, b1, tb) in zip(labelled, labelled[1:]):
        if ta == tb or b0 - a1 > settings.swing_gap_bins:
            continue
        peak_a = max(abs(series[i]["value"]) for i in range(a0, a1 + 1))
        peak_b = max(abs(series[i]["value"]) for i in range(b0, b1 + 1))
        total = peak_a + peak_b
        if total < settings.swing_total:
            continue
        mid = rows[(a1 + b0) // 2]["t"] + settings.bin_seconds / 2
        support = min(
            np.mean([series[i]["support"] for i in range(a0, b1 + 1)]), 1.0
        )
        moments.append(
            {
                "type": "momentum_swing",
                "t": mid,
                "start": rows[a1]["t"],
                "end": rows[b0]["t"] + settings.bin_seconds,
                "team": tb,
                "score": min(1.0, total / 1.2),
                "confidence": confidence_word(float(support), ball_share(rows, a0, b1)),
                "title": f"Momentum swings from {names[ta]} to {names[tb]}",
                "evidence": [
                    f"Momentum index reached {peak_a:.2f} in favour of {names[ta]}, then {peak_b:.2f} in favour of {names[tb]}, within {rows[b0]['t'] - rows[a1]['t']:.0f}s",
                    "Index blends observed ball control, players near the ball and territory where available",
                ],
            }
        )
    for sign, team in ((1, "A"), (-1, "B")):
        for start, end in runs(
            series, lambda v, s, sign=sign: sign * v >= settings.pressure_level and s >= 0.3
        ):
            length = end - start + 1
            if length < settings.pressure_min_bins:
                continue
            mean = float(np.mean([abs(series[i]["value"]) for i in range(start, end + 1)]))
            support = float(np.mean([series[i]["support"] for i in range(start, end + 1)]))
            moments.append(
                {
                    "type": "sustained_pressure",
                    "t": rows[start]["t"],
                    "start": rows[start]["t"],
                    "end": rows[end]["t"] + settings.bin_seconds,
                    "team": team,
                    "score": min(1.0, mean * min(1.0, length / 10) * 1.4),
                    "confidence": confidence_word(support, ball_share(rows, start, end)),
                    "title": f"{names[team]} sustain pressure for {length * settings.bin_seconds:.0f}s",
                    "evidence": [
                        f"Momentum index stayed at or beyond {settings.pressure_level:.2f} for {names[team]} (mean {mean:.2f})",
                    ],
                }
            )
    return moments


def event_moments(events, names):
    moments = []
    for event in events:
        if event.get("type") != "possession_change":
            continue
        low, high = event.get("uncertainty") or [None, None]
        moments.append(
            {
                "type": "possession_change",
                "t": event["timestamp"],
                "start": low if low is not None else event["timestamp"],
                "end": event.get("confirmed_at", event["timestamp"]),
                "team": event["to_team"],
                "score": 0.7,
                "confidence": "medium",
                "title": f"Control passes from {names[event['from_team']]} to {names[event['to_team']]}",
                "evidence": [
                    f"{event.get('observed_support_seconds', 0):.2f}s of continuous observed control by {len(event.get('support_track_ids', []))} nearby {names[event['to_team']]} player(s)",
                    "Ball observed next to the new team's players; colour-based team assignment",
                ],
            }
        )
    return moments


def zone_moments(frames, rows, settings, config, names):
    moments, current = [], None
    for record in frames + [None]:
        side = (
            in_box(record["anchor"], config)
            if record is not None and record["anchor"] is not None
            else None
        )
        if current and (
            record is None
            or side != current["side"]
            or record["t"] - current["last"] > settings.max_gap
        ):
            if current["last"] - current["start"] >= settings.box_min_seconds:
                moments.append(box_moment(current, settings, config, names))
            current = None
        if side and current is None:
            current = {
                "side": side,
                "start": record["t"],
                "last": record["t"],
                "kinds": [],
                "possession": [],
                "superiority": [],
            }
        if side and current:
            current["last"] = record["t"]
            current["kinds"].append(record["anchor_kind"])
            current["possession"].append(record["possession"])
            near = [p for p in record["players"] if math.dist(record["anchor"], p["xy"]) <= settings.focus_radius_m]
            current["superiority"].append(
                (sum(p["team"] == "A" for p in near), sum(p["team"] == "B" for p in near))
            )
    return moments


def box_moment(run, settings, config, names):
    duration = run["last"] - run["start"]
    owner = [p for p in run["possession"] if p in TEAMS]
    # A tie is no evidence either way: fall through to player superiority.
    team = (
        max(TEAMS, key=owner.count)
        if owner and owner.count("A") != owner.count("B")
        else None
    )
    source = "observed ball control"
    if team is None:
        counts = np.sum(np.asarray(run["superiority"]), axis=0)
        if counts[0] != counts[1]:
            team = "A" if counts[0] > counts[1] else "B"
            source = "more visible players near the ball"
    ball_share = run["kinds"].count("ball") / len(run["kinds"])
    where = "left" if run["side"] == "left" else "right"
    title = (
        f"Play inside the {where} penalty area for {duration:.1f}s"
        + (f", {names[team]} on top" if team else "")
    )
    evidence = [
        f"{'Ball' if ball_share >= 0.5 else 'Camera focus (proxy for the ball)'} inside the {where} penalty area for {duration:.1f}s ({ball_share:.0%} of samples had the ball observed)"
    ]
    if team:
        evidence.append(f"{names[team]} favoured by {source}")
    return {
        "type": "box_entry",
        "t": run["start"],
        "start": run["start"],
        "end": run["last"],
        "team": team,
        "score": min(1.0, 0.45 + duration / 8 + ball_share * 0.2),
        "confidence": "medium" if ball_share >= 0.5 else "low",
        "title": title,
        "evidence": evidence,
    }


def fast_ball_moments(ball_speeds, settings):
    moments, run = [], []
    for item in ball_speeds + [None]:
        # Above ~30 m/s is faster than any kick; treat it as a mapping glitch.
        if item is not None and settings.fast_ball_mps <= item[1] <= 30:
            # Samples either side of a missing-ball or shot gap are unrelated spikes.
            if run and item[0] - run[-1][0] > settings.max_gap:
                moments.extend(fast_ball_run(run, settings))
                run = []
            run.append(item)
            continue
        if run:
            moments.extend(fast_ball_run(run, settings))
            run = []
    return moments


def fast_ball_run(run, settings):
    peak = max(run, key=lambda r: r[1])
    if len(run) < 2 and peak[1] < settings.fast_ball_mps * 1.3:
        return []
    return [
        {
            "type": "fast_ball",
            "t": run[0][0],
            "start": run[0][0],
            "end": run[-1][0],
            "team": None,
            "score": min(1.0, 0.4 + (peak[1] - settings.fast_ball_mps) / 20),
            "confidence": "low",
            "title": "Fast ball movement (pass or shot candidate)",
            "evidence": [
                f"Observed ball moved at {peak[1]:.1f} m/s over consecutive samples",
                "Speed from detector positions; mapping noise can inflate it",
            ],
        }
    ]


def crowd_moments(frames, settings, names):
    moments, current = [], None
    for record in frames + [None]:
        crowd = None
        if record is not None and record["anchor"] is not None:
            near = [
                p
                for p in record["everyone"]
                if math.dist(record["anchor"], p["xy"]) <= settings.crowd_radius_m
            ]
            if len(near) >= settings.crowd_players:
                crowd = len(near)
        if current and (
            crowd is None or record["t"] - current["last"] > settings.max_gap
        ):
            if current["last"] - current["start"] >= settings.crowd_min_seconds:
                moments.append(
                    {
                        "type": "crowding",
                        "t": current["start"],
                        "start": current["start"],
                        "end": current["last"],
                        "team": None,
                        "score": min(1.0, 0.35 + max(current["counts"]) / 20),
                        "confidence": "medium" if current["kinds"].count("ball") * 2 >= len(current["kinds"]) else "low",
                        "title": f"{max(current['counts'])} players crowd the ball",
                        "evidence": [
                            f"{max(current['counts'])} observed players within {settings.crowd_radius_m:.0f} m of the {'ball' if current['kinds'].count('ball') * 2 >= len(current['kinds']) else 'camera focus'} for {current['last'] - current['start']:.1f}s",
                            "Typical of a duel, set piece or loose-ball scramble",
                        ],
                    }
                )
            current = None
        if crowd and current is None:
            current = {"start": record["t"], "last": record["t"], "counts": [], "kinds": []}
        if crowd and current:
            current["last"] = record["t"]
            current["counts"].append(crowd)
            current["kinds"].append(record["anchor_kind"])
    return moments


def tempo_moments(rows, settings):
    """Stretches where visible players move clearly faster than in the rest of the clip."""
    values = [r.get("intensity_mps") for r in rows]
    known = [v for v in values if v is not None]
    if len(known) < 8:
        return []
    median = float(np.median(known))
    threshold = max(median * settings.tempo_ratio, median + 0.5)
    moments, run = [], []
    for row, value in zip(rows + [{"t": None}], values + [None]):
        if value is not None and value >= threshold:
            run.append((row["t"], value))
            continue
        if len(run) >= settings.tempo_min_bins:
            mean = float(np.mean([v for _, v in run]))
            moments.append(
                {
                    "type": "tempo_surge",
                    "t": run[0][0],
                    "start": run[0][0],
                    "end": run[-1][0] + settings.bin_seconds,
                    "team": None,
                    "score": min(1.0, 0.3 + (mean / max(median, 0.5) - 1) * 1.2),
                    "confidence": "low",
                    "title": f"Tempo surge: visible players average {mean:.1f} m/s",
                    "evidence": [
                        f"Mean speed of observed players held at {mean:.1f} m/s for {len(run) * settings.bin_seconds:.0f}s against a clip median of {median:.1f} m/s",
                        "Speed from displacement over one second; broadcast mapping noise adds some",
                    ],
                }
            )
        run = []
    return moments


def select_moments(moments, settings):
    """Keep the strongest moments; merge near-duplicates of the same type."""
    ordered = sorted(moments, key=lambda m: -m["score"])
    kept = []
    for moment in ordered:
        if any(
            k["type"] == moment["type"] and abs(k["t"] - moment["t"]) < 4 for k in kept
        ):
            continue
        kept.append(moment)
        if len(kept) == settings.max_moments:
            break
    kept.sort(key=lambda m: m["t"])
    for index, moment in enumerate(kept):
        moment["id"] = index
        for key in ("t", "start", "end", "score"):
            moment[key] = round(float(moment[key]), 3)
    return kept


def shape_series(frames, settings):
    """Per-second visible-group shape for each team, with its support count."""
    per_second = defaultdict(lambda: {team: [] for team in TEAMS})
    anchors = defaultdict(list)
    for record in frames:
        if not record["eligible"]:
            continue
        second = int(record["t"])
        if record["focus"] is not None:
            anchors[second].append(record["focus"])
        for team in TEAMS:
            group = [p["xy"] for p in record["players"] if p["team"] == team]
            if len(group) >= settings.min_outfield:
                per_second[second][team].append(np.asarray(group))
    shapes = {}
    for second, teams in sorted(per_second.items()):
        anchor = (
            np.median(np.asarray(anchors[second]), axis=0)
            if anchors.get(second)
            else None
        )
        for team, groups in teams.items():
            if not groups:
                continue
            counts = [len(g) for g in groups]
            # Use the sample with the most players visible in this second.
            xy = groups[int(np.argmax(counts))]
            shapes.setdefault(team, {})[second] = {
                "n": len(xy),
                "cx": float(xy[:, 0].mean()),
                "cy": float(xy[:, 1].mean()),
                "width": float(np.ptp(xy[:, 1])),
                "depth": float(np.ptp(xy[:, 0])),
                "spread": float(np.sqrt(np.mean(np.sum((xy - xy.mean(axis=0)) ** 2, axis=1)))),
                "to_play": None
                if anchor is None
                else float(np.hypot(xy[:, 0].mean() - anchor[0], xy[:, 1].mean() - anchor[1])),
            }
    return shapes


def tactic_notes(shapes, rows, frames, settings, config, metadata, names):
    notes = []

    def add(kind, team, start, end, text, evidence, n):
        notes.append(
            {
                "kind": kind,
                "team": team,
                "start": float(start),
                "end": float(end) + 1.0,
                "text": text,
                "evidence": evidence,
                "players_observed": int(n),
            }
        )

    for team, by_second in shapes.items():
        seconds = sorted(by_second)
        depth = np.array([by_second[s]["depth"] for s in seconds])
        width = np.array([by_second[s]["width"] for s in seconds])
        if len(seconds) < 6:
            continue
        # A shape only counts as notable when it clearly departs from the
        # team's own median; near-constant shapes would otherwise tie.
        median = float(np.median(depth))
        low = (
            min(np.percentile(depth, 30), median - settings.min_shape_change),
            np.percentile(width, 30),
        )
        high = max(np.percentile(depth, 70), median + settings.min_shape_change)
        for kind, test in (
            ("compact", lambda s: by_second[s]["depth"] <= low[0] and by_second[s]["width"] <= low[1]),
            ("stretched", lambda s: by_second[s]["depth"] >= high),
        ):
            for run in consecutive(seconds, test):
                if len(run) < 3:
                    continue
                d = np.mean([by_second[s]["depth"] for s in run])
                w = np.mean([by_second[s]["width"] for s in run])
                n = int(np.median([by_second[s]["n"] for s in run]))
                if kind == "compact":
                    text = f"{names[team]} compact: visible group {d:.0f} m deep and {w:.0f} m wide, tighter than its usual shape in this clip"
                else:
                    text = f"{names[team]} stretched: visible group {d:.0f} m deep, longer than its usual {np.median(depth):.0f} m"
                add(kind, team, run[0], run[-1], text, {"depth_m": round(d, 1), "width_m": round(w, 1), "clip_median_depth_m": round(float(np.median(depth)), 1)}, n)
        # The visible group follows the camera, so absolute centroid movement
        # mostly reflects panning. Distance to the play is what changes tactically.
        window = int(settings.shift_window)

        def distance(second):
            found = [
                by_second[x]["to_play"]
                for x in (second - 1, second, second + 1)
                if x in by_second and by_second[x]["to_play"] is not None
            ]
            return float(np.median(found)) if len(found) >= 2 else None

        for s in seconds:
            start_d, end_d = distance(s), distance(s + window)
            if start_d is None or end_d is None:
                continue
            change = end_d - start_d
            n = min(by_second[s]["n"], by_second.get(s + window, by_second[s])["n"])
            evidence = {
                "distance_to_focus_start_m": round(start_d, 1),
                "distance_to_focus_end_m": round(end_d, 1),
                "seconds": window,
            }
            if change <= -settings.shift_metres:
                add("closing_in", team, s, s + window, f"{names[team]} close in: visible group's centre moved {abs(change):.0f} m closer to the play in {window}s", evidence, n)
            elif change >= settings.shift_metres:
                add("dropping_away", team, s, s + window, f"{names[team]} drop away: visible group's centre moved {change:.0f} m further from the play in {window}s", evidence, n)
    # Gap between the two visible groups.
    if all(t in shapes for t in TEAMS):
        shared = sorted(set(shapes["A"]) & set(shapes["B"]))
        gaps = {s: abs(shapes["A"][s]["cx"] - shapes["B"][s]["cx"]) for s in shared}
        for kind, test, text in (
            ("big_gap", lambda s: gaps[s] >= 20, "{g:.0f} m between the two visible groups: play is open"),
        ):
            for run in consecutive(shared, test):
                if len(run) < 3:
                    continue
                g = np.mean([gaps[s] for s in run])
                n = min(
                    int(np.median([shapes["A"][s]["n"] for s in run])),
                    int(np.median([shapes["B"][s]["n"] for s in run])),
                )
                add(kind, None, run[0], run[-1], text.format(g=g), {"centroid_gap_m": round(float(g), 1)}, n)
    # Numerical superiority around the ball or camera focus.
    key = {int(r["t"] // settings.bin_seconds): r for r in rows}
    for team, sign in (("A", 1), ("B", -1)):
        for run in consecutive(
            sorted(key),
            lambda k, sign=sign: key[k].get("near_anchor") is not None
            and sign * (key[k]["near_anchor"][0] - key[k]["near_anchor"][1]) >= 2,
        ):
            if len(run) < 2:
                continue
            near = np.mean([key[k]["near_anchor"] for k in run], axis=0)
            kinds = [key[k].get("anchor_kind") for k in run]
            where = "the ball" if kinds.count("ball") * 2 >= len(kinds) else "the camera focus"
            add(
                "overload",
                team,
                run[0] * settings.bin_seconds,
                run[-1] * settings.bin_seconds,
                f"{names[team]} outnumber the opposition among the six players nearest {where}: about {max(near):.0f} v {min(near):.0f}",
                {"near_A": round(float(near[0]), 1), "near_B": round(float(near[1]), 1)},
                int(round(near.sum())),
            )
    notes.sort(key=lambda n: (n["start"], n["kind"]))
    deduped, last = [], {}
    for note in notes:
        k = (note["kind"], note["team"])
        if k in last and note["start"] < last[k]["end"] + 2:
            last[k]["end"] = max(last[k]["end"], note["end"])
            continue
        deduped.append(note)
        last[k] = note
    for index, note in enumerate(deduped):
        note["id"] = index
        note["start"], note["end"] = round(note["start"], 2), round(note["end"], 2)
    return deduped


def consecutive(seconds, test):
    run = []
    for s in seconds:
        if test(s) and (not run or s - run[-1] <= 1):
            run.append(s)
            continue
        if run:
            yield run
        run = [s] if test(s) else []
    if run:
        yield run


def build(output, config, metadata, events, size=None, settings=None):
    """Return momentum, moments, tactic notes and coverage for one analysed run."""
    settings = replace(
        settings or Settings(),
        max_gap=2.5 / config.sample_hz,
        speed_tolerance=max(0.1, 0.6 / config.sample_hz),
    )
    names = names_of(metadata)
    frames, bins, ball_speeds = collect(output, config, metadata, size, settings)
    if not frames:
        raise ValueError("Run contains no sampled frames")
    directions_known = bool(metadata.get("attacking_directions"))
    rows = bin_series(bins, settings, metadata)
    series = smooth(rows, settings)
    moments = (
        momentum_moments(rows, series, settings, names)
        + event_moments(events, names)
        + zone_moments(frames, rows, settings, config, names)
        + fast_ball_moments(ball_speeds, settings)
        + crowd_moments(frames, settings, names)
        + tempo_moments(rows, settings)
    )
    shapes = shape_series(frames, settings)
    notes = tactic_notes(shapes, rows, frames, settings, config, metadata, names)
    eligible = [f for f in frames if f["eligible"]]
    coverage = {
        "sampled_frames": len(frames),
        "eligible_fraction": len(eligible) / len(frames),
        "ball_observed_fraction": sum(f["ball_raw"] is not None for f in eligible) / max(1, len(eligible)),
        "ball_usable_fraction": sum(f["ball"] is not None for f in eligible) / max(1, len(eligible)),
        "control_covered_fraction": sum(f["possession"] in TEAMS for f in eligible) / max(1, len(eligible)),
        "camera_focus_fraction": sum(f["anchor"] is not None for f in eligible) / max(1, len(eligible)),
        "mean_observed_players": float(np.mean([len(f["players"]) for f in eligible])) if eligible else 0.0,
        "momentum_supported_fraction": sum(s["value"] is not None for s in series) / len(series),
        "attacking_direction_known": directions_known,
    }
    momentum = []
    for row, smoothed in zip(rows, series):
        entry = {
            "t": round(row["t"], 3),
            "value": None if smoothed["value"] is None else round(smoothed["value"], 3),
            "support": round(smoothed["support"], 3),
            "raw": None if row["raw"] is None else round(row["raw"], 3),
            "components": row.get("components", {}),
        }
        for k in ("anchor", "anchor_kind", "near_anchor", "intensity_mps"):
            if k in row:
                entry[k] = [round(v, 1) for v in row[k]] if k == "anchor" else row[k]
        momentum.append(entry)
    selected = select_moments(moments, settings)
    for moment in selected:
        # Tactic notes that overlap a moment explain the shape behind it.
        moment["notes"] = [
            n["id"]
            for n in notes
            if n["start"] <= moment["end"] + 1 and n["end"] >= moment["start"] - 1
        ][:3]
    shape_rows, shape_stats = {}, {}
    for team, by_second in shapes.items():
        shape_rows[team] = [
            {
                "t": second,
                "n": item["n"],
                "width": round(item["width"], 1),
                "depth": round(item["depth"], 1),
                "cx": round(item["cx"], 1),
                "cy": round(item["cy"], 1),
            }
            for second, item in sorted(by_second.items())
        ]
        depth = np.array([item["depth"] for item in by_second.values()])
        width = np.array([item["width"] for item in by_second.values()])
        shape_stats[team] = {
            "seconds": len(depth),
            "depth_median": round(float(np.median(depth)), 1),
            "depth_low": round(float(np.percentile(depth, 30)), 1),
            "depth_high": round(float(np.percentile(depth, 70)), 1),
            "width_median": round(float(np.median(width)), 1),
        }
    return {
        "bin_seconds": settings.bin_seconds,
        "shapes": shape_rows,
        "shape_stats": shape_stats,
        "momentum": momentum,
        "moments": selected,
        "tactics": notes,
        "coverage": coverage,
    }
