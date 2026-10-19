"""Export a finished analysis directory as a static, self-contained web viewer payload."""

import colorsys
import json
import math
import shutil
import subprocess
from pathlib import Path

import cv2

from .config import Config
from .hashing import digest
from .insights import build
from .stats import player_stats, space_control
from .summary import load_frames

SCHEMA_VERSION = 1
KINDS = {"A": 0, "B": 1}
LIMITATIONS = [
    "Experimental: estimated spatial and control statistics, not official match statistics.",
    "Only the part of the pitch visible on the broadcast camera is observed; offscreen players are unknown.",
    "The momentum index blends observed ball control, players near the ball and territory; it is not a validated predictor.",
    "Where the ball is not detected, the camera focus stands in for the location of play.",
    "Teams are anonymous colour clusters unless names are supplied after verification.",
]


def team_colours(run):
    """Swatch colours from the accumulated jersey-colour cluster centres."""
    state = json.loads((run / "checkpoint.json").read_text()).get("team_state") or {}
    centres = state.get("centers")
    fallback = {"A": "#4da3ff", "B": "#ff6b6b"}
    if not centres:
        return fallback
    result = {}
    for team, (x, y, value) in zip("AB", centres):
        saturation = min(1.0, math.hypot(x, y))
        hue = (math.atan2(y, x) / (2 * math.pi)) % 1.0
        # Very dark or very pale kits would vanish against the pitch; keep them legible.
        red, green, blue = colorsys.hsv_to_rgb(
            hue, saturation, max(0.62, min(1.0, value))
        )
        result[team] = "#%02x%02x%02x" % (
            round(red * 255),
            round(green * 255),
            round(blue * 255),
        )
    return {**fallback, **result}


def person_row(person, shot):
    if person["role"] == "player":
        kind = KINDS.get(person.get("team"), 2)
    else:
        kind = 3 if person["role"] == "goalkeeper" else 4
    xy = person.get("xy")
    if xy is None:
        return None
    identity = shot * 10000 + int(person["id"].split(":")[-1])
    row = [
        identity,
        kind,
        round(xy[0], 1),
        round(xy[1], 1),
        1 if person["observation"] == "observed" else 0,
    ]
    if person.get("bbox"):
        row.extend(round(v) for v in person["bbox"])
    return row


def frame_payload(run):
    frames = []
    for frame in load_frames(run):
        rows = [
            row
            for person in frame["people"]
            if (row := person_row(person, frame["shot"])) is not None
        ]
        ball = frame["ball"]
        pixel = ball.get("pixel")
        matrix = (frame.get("calibration") or {}).get("homography")
        entry = {
            "t": round(frame["timestamp"], 3),
            "s": frame["shot"],
            "e": 1 if frame["eligible"] else 0,
            "p": rows,
            "b": None
            if ball.get("xy") is None
            else [
                round(ball["xy"][0], 1),
                round(ball["xy"][1], 1),
                1 if ball["observation"] == "observed" else 0,
                round(pixel[0]) if pixel else None,
                round(pixel[1]) if pixel else None,
                1 if ball.get("airborne_uncertain") else 0,
            ],
            "c": frame["possession"] if frame["possession"] in KINDS else None,
            # Image-to-pitch homography, used to draw the camera footprint.
            "h": None
            if matrix is None or not frame["eligible"]
            else [round(v, 7) for row in matrix for v in row],
        }
        frames.append(entry)
    return frames


def model_card(report, checkpoint_sha256, ball_from_other_model=False):
    """Validation figures for the detector, so the page states how good the model is and how that was measured."""
    report = json.loads(Path(report).read_text())
    if report.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(
            "Model report describes a different checkpoint than the one this run analysed"
        )
    validation = report["validation"]
    options = report.get("options", {})
    # A run that takes its ball from a separate adapted checkpoint did not get its ball from this detector.
    classes = [
        row for row in validation["per_class"] if not (ball_from_other_model and row["Class"] == "ball")
    ]
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "image_size": options.get("imgsz"),
        "split": "validation (used to choose the model; the test split is not used here)",
        "map50": round(validation["map50"], 3),
        "map50_95": round(validation["map50_95"], 3),
        "precision": round(validation["precision"], 3),
        "recall": round(validation["recall"], 3),
        "classes": [
            {
                "name": row["Class"],
                "instances": row["Instances"],
                "precision": round(row["Box-P"], 3),
                "recall": round(row["Box-R"], 3),
                "map50": round(row["mAP50"], 3),
            }
            for row in classes
        ],
    }


def transcode(video, output, max_height=720, duration=None):
    """Browser-friendly copies without audio, keeping original timestamps.

    H.264 plays in Safari, Chrome and Edge. Open-source Chromium builds and
    some Linux browsers lack H.264, so a VP9 WebM is offered alongside it.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to transcode the viewer video")
    scale = f"scale=-2:'min({max_height},ih)'"
    # Never publish footage beyond what was analysed (partial runs use --max-seconds).
    limit = ["-t", f"{duration:.3f}"] if duration else []
    common = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), *limit, "-an", "-vf", scale]
    subprocess.run(
        common
        + [
            "-c:v", "libx264", "-preset", "slow", "-crf", "27",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(output / "clip.mp4"),
        ],
        check=True,
    )
    subprocess.run(
        common
        + [
            "-c:v", "libvpx-vp9", "-crf", "36", "-b:v", "0", "-row-mt", "1",
            "-cpu-used", "4", "-pix_fmt", "yuv420p", str(output / "clip.webm"),
        ],
        check=True,
    )
    return [
        {"src": "clip.mp4", "type": "video/mp4"},
        {"src": "clip.webm", "type": "video/webm"},
    ]


def export_viewer(
    run, output, video=None, transcode_video=True, max_height=720, model_report=None
):
    run, output = Path(run).resolve(), Path(output).resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    signature = manifest["signature"]
    config = Config(**signature["config"])
    metadata = signature.get("metadata") or {}
    if not (run / "summary.json").is_file():
        raise ValueError("Run is incomplete: summary.json is missing")
    summary = json.loads((run / "summary.json").read_text())
    events = json.loads((run / "events.json").read_text())
    source_video = Path(video or manifest["video"])
    if not source_video.is_file():
        raise ValueError("Source video not found: " + str(source_video))
    # Tracks, timestamps and homographies belong to the analysed file only.
    if digest(source_video) != signature["video_sha256"]:
        raise ValueError("Video does not match the one this run analysed (SHA-256 differs)")
    capture = cv2.VideoCapture(str(source_video))
    size = [
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    ]
    fps = capture.get(cv2.CAP_PROP_FPS)
    length = capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0
    capture.release()
    partial = length > manifest["video_end"] + 0.25
    if not all(size):
        raise ValueError("Cannot read video dimensions")
    insights = build(run, config, metadata, events, size)
    output.mkdir(parents=True, exist_ok=True)
    video_name = "clip.mp4"
    if transcode_video:
        videos = transcode(
            source_video, output, max_height, manifest["video_end"] if partial else None
        )
    else:
        if partial:
            raise ValueError(
                "This run analysed only the first %.1f s; --no-transcode would publish the rest of the video"
                % manifest["video_end"]
            )
        shutil.copyfile(source_video, output / video_name)
        videos = [{"src": video_name, "type": "video/mp4"}]
    frames = frame_payload(run)
    limitations = list(LIMITATIONS)
    propagated = manifest.get("team_labels_propagated") or {}
    if propagated.get("enabled"):
        limitations.append(
            "Players whose jersey colour was unclear in some frames were given the team the same track shows "
            "elsewhere in the clip (%d player rows, by recompute-control --propagate-teams); the possession "
            "and momentum figures use those labels, so they rest on less direct colour evidence."
            % propagated.get("player_rows_relabelled", 0)
        )
    pitch = {"length": config.pitch_length, "width": config.pitch_width}
    duration = manifest["video_end"]
    extras = {
        "players": player_stats(frames, config.sample_hz, duration),
        "space_control": space_control(frames, pitch, size, duration, config.sample_hz),
    }
    if model_report:
        extras["model"] = model_card(
            model_report, signature["model_sha256"], bool(signature.get("ball_model_sha256"))
        )
    names = summary.get("team_names", {"A": "Team A", "B": "Team B"})
    colours = team_colours(run)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "experimental",
        "source": {
            "video": video_name,
            "videos": videos,
            "original_video_sha256": signature["video_sha256"],
            "model_sha256": signature["model_sha256"],
            "image_size": size,
            "sample_hz": config.sample_hz,
            "duration": manifest["video_end"],
            "hardware": manifest.get("runtime", {}).get("hardware"),
            "samples_per_compute_second": manifest.get(
                "processed_samples_per_compute_second"
            ),
        },
        "pitch": pitch,
        "teams": {
            team: {"name": names[team], "colour": colours[team]} for team in "AB"
        },
        "summary": {
            key: summary[key]
            for key in (
                "durations",
                "possession_coverage",
                "eligible_fraction",
                "observed_possession_percent",
                "possession_changes",
            )
        },
        "limitations": limitations,
        **insights,
        **extras,
        "frames": frames,
    }
    target = output / "analysis.json"
    target.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
    return {
        "analysis": str(target),
        "analysis_bytes": target.stat().st_size,
        "videos": {v["src"]: (output / v["src"]).stat().st_size for v in videos},
        "moments": len(insights["moments"]),
        "tactic_notes": len(insights["tactics"]),
        "coverage": insights["coverage"],
        "analysis_sha256": digest(target),
    }
