"""Per-player movement and nearest-player space control from the viewer frame payload.

Everything here is derived from the same pitch positions the viewer draws, so it
inherits their limits: only players on the broadcast camera are observed, and a
track can split into two ids when the tracker loses and re-finds a player.
"""

import math
from collections import Counter

import numpy as np

SMOOTH_SECONDS = 0.5
PEAK_SECONDS = 1.0
MAX_SPEED_MPS = 12.0  # faster than this between samples is an id switch or a calibration jump
HIGH_SPEED_MPS = 5.5  # 19.8 km/h
SPRINT_MPS = 7.0  # 25.2 km/h
MIN_TRACK_SECONDS = 2.0
GRID_STEP = 1.5  # metres
MIN_PLAYERS_PER_TEAM = 3
TEAM_OF_KIND = {0: "A", 1: "B"}
TEAM_AGREEMENT = 0.8


def _segments(samples, hz):
    """Split one track's observed samples wherever time, the camera shot or the raw position jumps.

    A raw step faster than any player can run is an identity switch or a calibration jump; it must
    split the segment *before* smoothing, otherwise the moving average spreads it into plausible-looking steps.
    """
    gap = 2.5 / hz
    segments, current = [], []
    for sample in samples:
        jump = False
        if current:
            dt = sample[0] - current[-1][0]
            jump = dt <= 0 or math.hypot(sample[1] - current[-1][1], sample[2] - current[-1][2]) / dt > MAX_SPEED_MPS
        if current and (sample[3] != current[-1][3] or sample[0] - current[-1][0] > gap or jump):
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


def _track_identity(kinds):
    """Goalkeeper, or an outfield player whose team is the one most of the track's assigned samples show."""
    if kinds[3] * 2 >= sum(kinds.values()):
        return "goalkeeper", None
    assigned = kinds[0] + kinds[1]
    if assigned and max(kinds[0], kinds[1]) / assigned >= TEAM_AGREEMENT:
        return "player", TEAM_OF_KIND[0 if kinds[0] > kinds[1] else 1]
    return "player", None


def player_stats(frames, hz, duration):
    """One row per tracked player (outfield or goalkeeper) with at least a few seconds observed.

    Samples where the jersey colour was unassigned still count towards a track's distance and time;
    the team is the one most of the track's assigned samples show, or none if it never settles.
    """
    tracks = {}
    for frame in frames:
        if not frame["e"]:
            continue
        for row in frame["p"]:
            if row[1] in (0, 1, 2, 3) and row[4]:
                track = tracks.setdefault(row[0], {"kinds": Counter(), "samples": []})
                track["kinds"][row[1]] += 1
                track["samples"].append((frame["t"], row[2], row[3], frame["s"]))
    step = 1.0 / hz
    seconds = int(math.ceil(duration))
    smooth = max(1, int(round(SMOOTH_SECONDS * hz)))
    window = max(1, int(round(PEAK_SECONDS * hz)))
    result = []
    for identity, track in tracks.items():
        distance = observed = high = sprint = 0.0
        peak = None
        bins = [[] for _ in range(seconds)]
        for segment in _segments(track["samples"], hz):
            if len(segment) < 3:
                continue
            times = np.array([s[0] for s in segment])
            xs = _smooth(np.array([s[1] for s in segment]), smooth)
            ys = _smooth(np.array([s[2] for s in segment]), smooth)
            observed += times[-1] - times[0] + step
            dt = np.diff(times)
            speed = np.hypot(np.diff(xs), np.diff(ys)) / np.where(dt > 0, dt, np.inf)
            distance += float(np.sum(speed * dt))
            # A one-second running mean keeps a single jittery step from becoming the "top speed".
            running = _smooth(speed, window) if len(speed) >= window else speed
            if len(speed) >= window:
                peak = max(peak or 0.0, float(running.max()))
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
        role, team = _track_identity(track["kinds"])
        result.append(
            {
                "id": identity,
                "team": team,
                "role": role,
                "seconds": round(float(observed), 1),
                "distance_m": round(float(distance)),
                "avg_kmh": round(float(distance / observed * 3.6), 1),
                # None until some segment lasts a full peak window: a shorter one would only guess.
                "peak_kmh": None if peak is None else round(float(peak * 3.6), 1),
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


def space_control(frames, pitch, size, duration, hz=10.0, step=GRID_STEP):
    """Share of the camera-visible pitch closer to each team's observed outfield players.

    Per second, averaged over that second's supported frames, with `support` the fraction of the
    second's expected samples that qualified. `A` is `None` where no frame had three observed
    players per team in a mappable view. The headline mean and `supported_fraction` are weighted by
    supported sample time, so a second with one qualifying frame counts a tenth of a full one.
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
                    "support": round(min(1.0, len(samples) / hz), 2),
                }
            )
        else:
            series.append({"t": t, "A": None, "area_m2": 0, "support": 0.0})
    weights = [len(samples) for samples in bins]
    total = sum(weights)
    mean = sum(w * float(np.mean([x[0] for x in samples])) for w, samples in zip(weights, bins) if samples) / total if total else None
    return {
        "method": "nearest observed outfield player, camera-visible pitch only",
        "series": series,
        "mean_A": None if mean is None else round(mean, 3),
        "supported_fraction": round(min(1.0, total / (hz * max(1, duration))), 3),
    }
