"""Per-player movement and nearest-player space control from the viewer frame payload.

Everything here is derived from the same pitch positions the viewer draws, so it
inherits their limits: only players on the broadcast camera are observed, and a
track can split into two ids when the tracker loses and re-finds a player.
"""

import math

import numpy as np

SMOOTH_SAMPLES = 5  # 0.5 s at 10 Hz
MAX_SPEED_MPS = 12.0  # faster than this between samples is an id switch or a calibration jump
HIGH_SPEED_MPS = 5.5  # 19.8 km/h
SPRINT_MPS = 7.0  # 25.2 km/h
MIN_TRACK_SECONDS = 2.0
GRID_STEP = 1.5  # metres
MIN_PLAYERS_PER_TEAM = 3
TEAM_OF_KIND = {0: "A", 1: "B"}


def _segments(samples, hz):
    """Split one track's observed samples wherever time or the camera shot jumps."""
    gap = 2.5 / hz
    segments, current = [], []
    for sample in samples:
        if current and (sample[3] != current[-1][3] or sample[0] - current[-1][0] > gap):
            segments.append(current)
            current = []
        current.append(sample)
    if current:
        segments.append(current)
    return segments


def _smooth(values, width):
    if len(values) < width:
        return values
    kernel = np.ones(width) / width
    padded = np.pad(values, (width // 2, width - 1 - width // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def player_stats(frames, hz, duration):
    """One row per tracked player (outfield or goalkeeper) with at least a few seconds observed."""
    tracks = {}
    for frame in frames:
        if not frame["e"]:
            continue
        for row in frame["p"]:
            if row[1] in (0, 1, 3) and row[4]:
                tracks.setdefault(row[0], {"kind": row[1], "samples": []})["samples"].append(
                    (frame["t"], row[2], row[3], frame["s"])
                )
    step = 1.0 / hz
    seconds = int(math.ceil(duration))
    result = []
    for identity, track in tracks.items():
        distance = observed = high = sprint = 0.0
        peak = 0.0
        bins = [[] for _ in range(seconds)]
        for segment in _segments(track["samples"], hz):
            if len(segment) < 3:
                continue
            times = np.array([s[0] for s in segment])
            xs = _smooth(np.array([s[1] for s in segment]), SMOOTH_SAMPLES)
            ys = _smooth(np.array([s[2] for s in segment]), SMOOTH_SAMPLES)
            observed += times[-1] - times[0] + step
            dt = np.diff(times)
            speed = np.hypot(np.diff(xs), np.diff(ys)) / np.where(dt > 0, dt, np.inf)
            valid = speed <= MAX_SPEED_MPS
            speed = np.where(valid, speed, 0.0)
            distance += float(np.sum(speed * dt))
            # A one-second running mean keeps a single jittery step from becoming the "top speed".
            window = max(1, int(round(hz)))
            running = _smooth(speed, window) if len(speed) >= window else speed
            if len(speed) >= window:
                peak = max(peak, float(running.max()))
            high += float(np.sum(dt[running >= HIGH_SPEED_MPS]))
            sprint += float(np.sum(dt[running >= SPRINT_MPS]))
            for t, value in zip(times[1:], running):
                index = min(seconds - 1, int(t))
                if 0 <= index < seconds:
                    bins[index].append(float(value))
        if observed < MIN_TRACK_SECONDS:
            continue
        per_second = [round(float(np.mean(b)) * 3.6, 1) if b else None for b in bins]
        speeds = [v for v in per_second if v is not None]
        result.append(
            {
                "id": identity,
                "team": TEAM_OF_KIND.get(track["kind"]),
                "role": "goalkeeper" if track["kind"] == 3 else "player",
                "seconds": round(float(observed), 1),
                "distance_m": round(float(distance)),
                "avg_kmh": round(float(distance / observed * 3.6), 1),
                "peak_kmh": round(float(peak * 3.6), 1),
                "high_speed_s": round(float(high), 1),
                "sprint_s": round(float(sprint), 1),
                "first": round(track["samples"][0][0], 1),
                "last": round(track["samples"][-1][0], 1),
                "kmh_per_s": per_second if speeds else [],
            }
        )
    result.sort(key=lambda r: (-r["distance_m"], r["id"]))
    return result


def _visible_cells(matrix, size, pitch, step):
    """Pitch cell centres the camera sees, from the image-to-pitch homography."""
    h = np.array(matrix, dtype=float).reshape(3, 3)
    try:
        inverse = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        return None
    xs = np.arange(step / 2, pitch["length"], step)
    ys = np.arange(step / 2, pitch["width"], step)
    gx, gy = np.meshgrid(xs, ys)
    points = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)])
    image = inverse @ points
    w = image[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = image[0] / w, image[1] / w
    mask = (w > 0) & (u >= 0) & (u < size[0]) & (v >= 0) & (v < size[1])
    return np.stack([gx.ravel()[mask], gy.ravel()[mask]], axis=1)


def space_control(frames, pitch, size, duration, step=GRID_STEP):
    """Share of the camera-visible pitch closer to each team's observed outfield players.

    Per second, averaged over that second's supported frames. `None` where fewer than
    three observed players per team are available or the view is not mappable.
    """
    seconds = int(math.ceil(duration))
    bins = [[] for _ in range(seconds)]
    for frame in frames:
        if not frame["e"] or frame["h"] is None:
            continue
        a = [(r[2], r[3]) for r in frame["p"] if r[1] == 0 and r[4]]
        b = [(r[2], r[3]) for r in frame["p"] if r[1] == 1 and r[4]]
        if len(a) < MIN_PLAYERS_PER_TEAM or len(b) < MIN_PLAYERS_PER_TEAM:
            continue
        cells = _visible_cells(frame["h"], size, pitch, step)
        if cells is None or len(cells) < 20:
            continue
        pa, pb = np.array(a), np.array(b)
        da = np.min(np.hypot(cells[:, None, 0] - pa[None, :, 0], cells[:, None, 1] - pa[None, :, 1]), axis=1)
        db = np.min(np.hypot(cells[:, None, 0] - pb[None, :, 0], cells[:, None, 1] - pb[None, :, 1]), axis=1)
        index = min(seconds - 1, int(frame["t"]))
        bins[index].append((float(np.mean(da < db)), len(cells) * step * step))
    series = []
    for t, samples in enumerate(bins):
        if samples:
            series.append(
                {
                    "t": t,
                    "A": round(float(np.mean([s[0] for s in samples])), 3),
                    "area_m2": round(float(np.mean([s[1] for s in samples]))),
                }
            )
        else:
            series.append({"t": t, "A": None, "area_m2": 0})
    supported = [s["A"] for s in series if s["A"] is not None]
    return {
        "method": "nearest observed outfield player, camera-visible pitch only",
        "series": series,
        "mean_A": round(float(np.mean(supported)), 3) if supported else None,
        "supported_fraction": round(len(supported) / max(1, seconds), 3),
    }
