"""Automatic team-control evidence export and CPU-only artifact reprocessing."""

import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from .config import Config
from .hashing import digest
from .state import Possession
from .summary import load_frames, summarize


def export(output, config, video_end):
    output = Path(output)
    reasons = Counter()
    ball_run = longest_ball = 0
    prior = None
    frames = iter(load_frames(output))
    frame = next(frames, None)
    with (output / "possession.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "timestamp",
                "shot",
                "team",
                "candidate_team",
                "reason",
                "observed_support_seconds",
                "nearest_distance_m",
                "competing_distance_m",
                "support_track_ids",
                "ball_observation",
                "ball_reason",
            ],
        )
        writer.writeheader()
        while frame is not None:
            following = next(frames, None)
            end = following["timestamp"] if following else video_end
            dt = max(0, min(end - frame["timestamp"], 2 / config.sample_hz))
            evidence = frame.get("control_evidence", {})
            reason = evidence.get("reason", "legacy_unrecorded")
            reasons[reason] += dt
            observed = frame["eligible"] and frame["ball"]["observation"] == "observed"
            continuous = (
                prior is not None
                and prior["shot"] == frame["shot"]
                and 0 < frame["timestamp"] - prior["timestamp"] <= 0.25
            )
            ball_run = (ball_run if continuous else 0) + dt if observed else 0
            longest_ball = max(longest_ball, ball_run)
            writer.writerow(
                {
                    "timestamp": frame["timestamp"],
                    "shot": frame["shot"],
                    "team": frame["possession"],
                    "reason": reason,
                    **{
                        k: evidence.get(k)
                        for k in (
                            "candidate_team",
                            "observed_support_seconds",
                            "nearest_distance_m",
                            "competing_distance_m",
                        )
                    },
                    "support_track_ids": json.dumps(
                        evidence.get("support_track_ids", [])
                    ),
                    "ball_observation": frame["ball"]["observation"],
                    "ball_reason": frame["ball"].get("reason"),
                }
            )
            prior, frame = frame, following
    report = {
        "schema_version": 1,
        "runtime_human_input_required": False,
        "method": "Compare stable observed controlling-team colour assignments; teammates do not create team ambiguity.",
        "reason_seconds": dict(reasons),
        "longest_contiguous_observed_ball_seconds": longest_ball,
        "required_control_persistence_seconds": config.possession_persistence,
        "accuracy_validated": False,
    }
    (output / "possession_diagnostics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def track_teams(source, min_samples=5, agreement=0.8):
    """Team each player track shows for most of a run, keyed by (shot, track id).

    Team colours are learned online, so a player can be unassigned for some frames even
    though the same track is clearly one team elsewhere. A track needs `min_samples`
    assigned observations with `agreement` of them on one team.
    """
    votes = defaultdict(Counter)
    for shot_file in sorted((Path(source) / "shots").glob("*.jsonl")):
        with shot_file.open() as lines:
            for line in lines:
                frame = json.loads(line)
                for person in frame["people"]:
                    if person["role"] == "player" and person.get("team") in ("A", "B"):
                        votes[(frame["shot"], person["id"])][person["team"]] += 1
    teams = {}
    for key, counter in votes.items():
        total = sum(counter.values())
        team, count = counter.most_common(1)[0]
        if total >= min_samples and count / total >= agreement:
            teams[key] = team
    return teams


def recompute(source, output, refresh_airborne=False, propagate_teams=False):
    """Keep detector/calibration observations intact and replace control inference.

    Preserve source signatures/hashes separately. This is not a new GPU benchmark,
    new detector evaluation, or independent ground truth.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Preserve existing outputs; choose a new directory")
    if not json.loads((source / "checkpoint.json").read_text()).get("complete"):
        raise ValueError("Reprocessing requires a completed source run")
    manifest = json.loads((source / "manifest.json").read_text())
    config = Config(**manifest["signature"]["config"])
    controller = Possession(
        config.possession_persistence,
        config.possession_radius_m,
        config.ambiguity_margin_m,
    )
    provenance = {
        str(p.relative_to(source)): digest(p)
        for p in [
            source / "manifest.json",
            source / "checkpoint.json",
            *sorted((source / "shots").glob("*.jsonl")),
        ]
    }
    teams = track_teams(source) if propagate_teams else {}
    relabelled = 0
    output.mkdir(parents=True)
    (output / "shots").mkdir()
    (output / "checkpoint.json").write_text(json.dumps({"complete": False}))
    started = time.perf_counter()
    prior_shot = None
    prior_time = -1
    count = 0
    for shot_file in sorted((source / "shots").glob("*.jsonl")):
        with (
            shot_file.open() as src,
            (output / "shots" / shot_file.name).open("w") as dst,
        ):
            for line in src:
                frame = json.loads(line)
                if frame["timestamp"] <= prior_time:
                    raise ValueError("Source timestamps must increase")
                if frame["shot"] != prior_shot:
                    controller.reset()
                prior_shot, prior_time = frame["shot"], frame["timestamp"]
                for person in frame["people"] if teams else ():
                    team = teams.get((frame["shot"], person["id"]))
                    if (
                        person["role"] == "player"
                        and person.get("team") not in ("A", "B")
                        and team
                    ):
                        person["team"] = team
                        relabelled += 1
                ball = frame["ball"]
                if (
                    refresh_airborne
                    and ball.get("pixel")
                    and ball["observation"] == "observed"
                ):
                    from .vision import airborne_uncertainty

                    ball["airborne_uncertain"] = airborne_uncertainty(
                        ball["pixel"], frame["people"]
                    )
                team, event = controller.update(
                    frame["timestamp"],
                    ball.get("xy") if ball["observation"] == "observed" else None,
                    frame["people"],
                    frame["eligible"],
                    ball.get("airborne_uncertain", False),
                )
                if event:
                    event["shot"] = frame["shot"]
                frame.update(
                    possession=team, event=event, control_evidence=controller.evidence
                )
                dst.write(json.dumps(frame, allow_nan=False) + "\n")
                count += 1
    result = summarize(
        output, config, manifest["signature"].get("metadata", {}), manifest["video_end"]
    )
    from .coaching import export as coaching_export

    coaching_export(output, config)
    export(output, config, manifest["video_end"])
    derived = {
        **manifest,
        "analysis_stage": "automatic_control_reprocessing",
        "airborne_heuristic_refreshed": refresh_airborne,
        "team_labels_propagated": {"enabled": propagate_teams, "player_rows_relabelled": relabelled},
        "source_run": str(source),
        "source_artifact_sha256": provenance,
        "source_detection_calibration_signature": manifest["signature"],
        "control_code_sha256": {
            p.name: digest(p)
            for p in (
                Path(__file__),
                Path(__file__).with_name("state.py"),
                Path(__file__).with_name("vision.py"),
            )
        },
        "compute_seconds": time.perf_counter() - started,
        "processed_samples": count,
        "processed_samples_per_compute_second": None,
        "throughput_scope": "CPU control reprocessing only; not detector/calibration or cloud throughput",
    }
    (output / "manifest.json").write_text(
        json.dumps(derived, indent=2, allow_nan=False) + "\n"
    )
    # Keep the learned jersey colours: the viewer export draws the team swatches from them.
    source_state = json.loads((source / "checkpoint.json").read_text()).get("team_state")
    (output / "checkpoint.json").write_text(
        json.dumps({"complete": True, "team_state": source_state}) + "\n"
    )
    return result
