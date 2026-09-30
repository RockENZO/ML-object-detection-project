"""Frozen match partitions and independently annotated, observed-only evaluation."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .hashing import digest
from .summary import load_frames


def freeze(sources, output):
    output = Path(output)
    if output.exists():
        raise ValueError("A frozen partition cannot be overwritten")
    records = json.loads(Path(sources).read_text())
    if not records or any(
        not r.get("match_id") or not r.get("clip_id") for r in records
    ):
        raise ValueError("Every clip needs verified source match_id and clip_id")
    if len({r["clip_id"] for r in records}) != len(records):
        raise ValueError("Duplicate clip IDs")
    # Stable hashing means adding clips from an existing match cannot change its split.
    assigned_partitions = {}
    for record in records:
        choice = record.get("requested_partition")
        if choice is not None and choice not in (
            "development",
            "validation",
            "evaluation",
        ):
            raise ValueError("Invalid requested partition")
        if choice is not None:
            previous = assigned_partitions.get(record["match_id"])
            if previous is not None and previous != choice:
                raise ValueError("One match cannot span partitions")
            assigned_partitions[record["match_id"]] = choice
    for record in records:
        value = (
            int(
                hashlib.sha256(
                    ("football-v1:" + record["match_id"]).encode()
                ).hexdigest()[:8],
                16,
            )
            % 10
        )
        record["partition"] = (
            "development" if value < 6 else "validation" if value < 8 else "evaluation"
        )
        record["partition"] = assigned_partitions.get(
            record["match_id"], record["partition"]
        )
        for field in ("video", "annotations"):
            if field in record:
                record[field] = str(Path(record[field]).resolve())
                record[field + "_sha256"] = digest(record[field])
    result = {
        "schema_version": 1,
        "seed": "football-v1",
        "match_overrides": assigned_partitions,
        "sources": records,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def iou(a, b):
    x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - x * y
    return x * y / union if union > 0 else 0


def match_boxes(truth, predicted, role_sensitive=True):
    if not truth or not predicted:
        return []
    scores = np.array(
        [
            [
                iou(a["bbox"], b["bbox"])
                if not role_sensitive or a["role"] == b["role"]
                else 0
                for b in predicted
            ]
            for a in truth
        ]
    )
    rows, cols = linear_sum_assignment(-scores)
    return [(int(a), int(b)) for a, b in zip(rows, cols) if scores[a, b] >= 0.5]


def transition_metrics(truth, predicted, tolerance=1.0):
    if truth and predicted:
        cost = np.full((len(truth), len(predicted)), 1e6)
        for i, a in enumerate(truth):
            for j, b in enumerate(predicted):
                error = abs(a["timestamp"] - b["timestamp"])
                if error <= tolerance and (a["from_team"], a["to_team"]) == (
                    b["from_team"],
                    b["to_team"],
                ):
                    cost[i, j] = error
        rows, cols = linear_sum_assignment(cost)
        errors = [float(cost[i, j]) for i, j in zip(rows, cols) if cost[i, j] < 1e6]
    else:
        errors = []
    tp = len(errors)
    return {
        "precision": tp / len(predicted) if predicted else None,
        "recall": tp / len(truth) if truth else None,
        "true_positives": tp,
        "false_positives": len(predicted) - tp,
        "false_negatives": len(truth) - tp,
        "timestamp_mae_seconds": float(np.mean(errors)) if errors else None,
        "tolerance_seconds": tolerance,
    }


def evaluate(run, annotations, partition, output):
    run = Path(run)
    output = Path(output)
    gt = json.loads(Path(annotations).read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    frozen = json.loads(Path(partition).read_text())
    clip = gt.get("clip_id")
    source = next((r for r in frozen["sources"] if r["clip_id"] == clip), None)
    if source is None or source.get("match_id") != gt.get("match_id"):
        raise ValueError("Ground truth is not in frozen source matches")
    if source.get("video_sha256") != manifest["signature"][
        "video_sha256"
    ] or source.get("annotations_sha256") != digest(annotations):
        raise ValueError("Video or annotation hash does not match frozen partition")
    if output.exists():
        raise ValueError("Evaluation output exists; preserve the previous result")
    if not json.loads((run / "checkpoint.json").read_text())["complete"]:
        raise ValueError("Finish or resume analysis before evaluating")
    frames = {f["frame_index"]: f for f in load_frames(run)}
    pairs = Counter()
    role_pairs = Counter()
    role_correct = role_matches = 0
    ngt = npred = assigned = teamright = teamtotal = ballgt = ballpred = balltp = 0
    raw_ball_gt = raw_ball_pred = raw_ball_tp = 0
    position = []
    segmentation = Counter()
    covered = eligible_truth = 0
    sampled = []
    # Team A/B mapping must be recorded independently, before freezing annotations.
    mapping = gt.get("team_mapping", {"A": "A", "B": "B"})
    team_pairs = Counter()
    pitch_possible = 0
    truth_frames = gt.get("frames", [])
    if len({f["frame_index"] for f in truth_frames}) != len(truth_frames):
        raise ValueError("Duplicate annotation frames")
    for truth in truth_frames:
        index = truth["frame_index"]
        if index not in frames:
            raise ValueError(
                f"Annotated frame {index} was not analyzed; annotate the sample grid"
            )
        prediction = frames[index]
        sampled.append(prediction)
        actual = truth.get("people", [])
        observed = [
            p
            for p in prediction["people"]
            if p["observation"] == "observed" and p.get("bbox")
        ]
        if "people" in truth:
            ngt += len(actual)
            npred += len(observed)
            for a in actual:
                for b in observed:
                    if iou(a["bbox"], b["bbox"]) >= 0.5:
                        if a["role"] == b["role"]:
                            role_pairs[(str(a["id"]), b["id"])] += 1
                        pairs[(str(a["id"]), b["id"])] += 1
            for i, j in match_boxes(actual, observed, role_sensitive=False):
                a, b = actual[i], observed[j]
                role_matches += 1
                role_correct += a["role"] == b["role"]
                if a["role"] == "player" and a.get("xy") is not None:
                    pitch_possible += 1
                if a.get("team") and a["role"] == "player":
                    teamtotal += 1
                    if b.get("team"):
                        assigned += 1
                        team_pairs[(b["team"], a["team"])] += 1
                        teamright += mapping[b["team"]] == a["team"]
                if (
                    a["role"] == "player"
                    and a.get("xy") is not None
                    and b.get("xy") is not None
                ):
                    position.append(math.dist(a["xy"], b["xy"]))
        if "ball" in truth:
            a = truth["ball"]
            raw = prediction.get("ball_detections", [])
            raw_ball_gt += bool(a and a.get("bbox"))
            raw_ball_pred += len(raw)
            raw_ball_tp += len(
                match_boxes([{**a, "role": "ball"}] if a and a.get("bbox") else [], raw)
            )
            b = (
                prediction["ball"]
                if prediction["ball"]["observation"] == "observed"
                else None
            )
            ballgt += bool(a and a.get("bbox"))
            ballpred += bool(b and b.get("bbox"))
            balltp += bool(
                a
                and b
                and a.get("bbox")
                and b.get("bbox")
                and iou(a["bbox"], b["bbox"]) >= 0.5
            )
        if "eligible" in truth:
            segmentation[
                f"eligible_truth_{int(truth['eligible'])}_pred_{int(prediction['view']['eligible_view'])}"
            ] += 1
        if "replay" in truth:
            segmentation[
                f"replay_truth_{int(truth['replay'])}_pred_{int(prediction['view']['view'] == 'replay_candidate')}"
            ] += 1
        if truth.get("eligible"):
            eligible_truth += 1
            covered += prediction["possession"] in ("A", "B")
    if gt.get("team_label_permutation_invariant") and team_pairs:
        labels = sorted({team for _, team in team_pairs})
        if len(labels) == 2:
            variants = [dict(zip(("A", "B"), labels)), dict(zip(("B", "A"), labels))]
            mapping = max(
                variants,
                key=lambda m: sum(n for (p, g), n in team_pairs.items() if m[p] == g),
            )
            teamright = sum(n for (p, g), n in team_pairs.items() if mapping[p] == g)
    ids = sorted({a for a, b in pairs})
    tracks = sorted({b for a, b in pairs})
    idtp = 0
    if ids and tracks:
        counts = np.array([[pairs[(a, b)] for b in tracks] for a in ids])
        rows, cols = linear_sum_assignment(-counts)
        idtp = int(counts[rows, cols].sum())
    idf1 = 2 * idtp / (ngt + npred) if ngt + npred else None
    role_idtp = 0
    if ids and tracks:
        role_counts = np.array([[role_pairs[(a, b)] for b in tracks] for a in ids])
        rows, cols = linear_sum_assignment(-role_counts)
        role_idtp = int(role_counts[rows, cols].sum())
    predicted_events = json.loads((run / "events.json").read_text())
    if gt.get("transitions_use_ground_truth_team_labels"):
        predicted_events = [
            {
                **e,
                "from_team": mapping.get(e["from_team"]),
                "to_team": mapping.get(e["to_team"]),
            }
            for e in predicted_events
        ]
    changes = (
        transition_metrics(gt["transitions"], predicted_events)
        if "transitions" in gt
        else None
    )
    metrics = {
        "tracking_idf1": idf1,
        "role_sensitive_tracking_idf1": 2 * role_idtp / (ngt + npred)
        if ngt + npred
        else None,
        "role_accuracy_on_matches": role_correct / role_matches
        if role_matches
        else None,
        "identity_counts": {"IDTP": idtp, "IDFN": ngt - idtp, "IDFP": npred - idtp},
        "team_accuracy_on_assigned": teamright / assigned if assigned else None,
        "team_assignment_coverage_on_matches": assigned / teamtotal
        if teamtotal
        else None,
        "median_player_pitch_error_metres": float(np.median(position))
        if position
        else None,
        "pitch_matched_observations": len(position),
        "pitch_projection_coverage_on_matches": len(position) / pitch_possible
        if pitch_possible
        else None,
        "team_mapping": mapping,
        "raw_ball_precision": raw_ball_tp / raw_ball_pred if raw_ball_pred else None,
        "raw_ball_recall": raw_ball_tp / raw_ball_gt if raw_ball_gt else None,
        "ball_precision": balltp / ballpred if ballpred else None,
        "ball_recall": balltp / ballgt if ballgt else None,
        "possession_coverage_on_eligible_annotations": covered / eligible_truth
        if eligible_truth
        else None,
        "segmentation_confusion_counts": dict(segmentation),
        "transitions": changes,
    }
    targets = {
        "tracking_idf1": 0.75,
        "team_accuracy_on_assigned": 0.9,
        "median_player_pitch_error_metres": 2.0,
    }
    gates = {
        k: metrics[k] is not None
        and (metrics[k] <= v if "error" in k else metrics[k] >= v)
        for k, v in targets.items()
    }
    gates["transition_precision"] = bool(
        changes and changes["precision"] is not None and changes["precision"] >= 0.85
    )
    gates["transition_recall"] = bool(
        changes and changes["recall"] is not None and changes["recall"] >= 0.70
    )
    result = {
        "status": "experimental",
        "partition": source["partition"],
        "clip_id": clip,
        "match_id": source["match_id"],
        "evaluation_code_sha256": digest(__file__),
        "annotation_sha256": digest(annotations),
        "partition_sha256": digest(partition),
        "run_signature": manifest["signature"],
        "annotated_frames": len(sampled),
        "metrics": metrics,
        "release_targets_met": gates,
        "limitations": [
            "Person IDF1 uses geometry at IoU >=0.5; role-sensitive legacy IDF1 and role accuracy are reported separately. Not official GS-HOTA.",
            "A single clip cannot certify a full release; aggregate held-out source matches and inspect failures.",
            "Missing labels remain unmeasured; ball-action labels do not imply possession-change ground truth.",
        ],
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
