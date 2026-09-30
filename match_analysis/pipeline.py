"""Streaming detector/calibration pipeline with atomic shot-level recovery."""

import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from prepare_grouped_split import NAMES

from .calibration import PitchMapper
from .config import Config
from .hashing import digest
from .state import Motion, Possession, TeamAssigner
from .summary import summarize, validate_metadata
from .vision import Segmenter, colour


def atomic_json(path, value):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


class Detector:
    def __init__(self, checkpoint, device, config):
        import ultralytics
        import yaml
        from ultralytics import YOLO
        from ultralytics.trackers.bot_sort import BOTSORT

        self.model = YOLO(str(checkpoint))
        if list(self.model.names.values()) != NAMES:
            raise ValueError("A trained four-class football checkpoint is required")
        self.device, self.config = device, config
        import torch

        self.hardware = (
            torch.cuda.get_device_name(int(str(device).split(":")[-1]))
            if str(device).isdigit() or str(device).startswith("cuda:")
            else platform.machine()
        )
        args = yaml.safe_load(
            (
                Path(ultralytics.__file__).parent / "cfg/trackers/botsort.yaml"
            ).read_text()
        )
        self.tracker = BOTSORT(SimpleNamespace(**args), frame_rate=config.sample_hz)

    def reset(self):
        self.tracker.reset()

    def predict(self, frame):
        return self.model.predict(
            frame, imgsz=self.config.imgsz, device=self.device, conf=0.15, verbose=False
        )[0]

    def track(self, detection, frame):
        people = detection.boxes[detection.boxes.cls != 0].cpu().numpy()
        return self.tracker.update(people, frame)


def analyze(
    video,
    checkpoint,
    output,
    config=None,
    metadata=None,
    pitch_root=None,
    device="cpu",
    resume=False,
    max_seconds=None,
    detector=None,
    mapper=None,
):
    config = config or Config()
    metadata = validate_metadata(metadata or {})
    video, checkpoint, output = (
        Path(video).resolve(),
        Path(checkpoint).resolve(),
        Path(output).resolve(),
    )
    if not video.is_file() or not checkpoint.is_file():
        raise ValueError("Video and model must exist")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError("Cannot decode video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or count <= 0:
        cap.release()
        raise ValueError("Video metadata is invalid")
    video_end = (
        min(count / fps, max_seconds) if max_seconds is not None else count / fps
    )
    if video_end <= 0:
        cap.release()
        raise ValueError("Analysis duration must be positive")

    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    packages = {
        name: version(name)
        for name in (
            "torch",
            "torchvision",
            "ultralytics",
            "numpy",
            "scipy",
            "opencv-python",
            "lap",
            "shapely",
            "lsq-ellipse",
        )
    }
    signature = {
        "runtime_versions": packages,
        "video_sha256": digest(video),
        "model_sha256": digest(checkpoint),
        "config": config.as_dict(),
        "metadata": metadata,
        "max_seconds": max_seconds,
        "device": device,
        "code_hashes": {
            p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        "pitch_assets_sha256": digest(Path(__file__).parent / "pnl_assets.json"),
    }
    existed = output.exists()
    if existed and not resume:
        cap.release()
        raise ValueError("Output exists; use --resume or a new directory")
    if resume and not existed:
        cap.release()
        raise ValueError("Cannot resume a missing output directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "shots").mkdir(exist_ok=True)
    state = {
        "next_frame": 0,
        "next_due": 0.0,
        "shot": 0,
        "team_state": None,
        "replay_history": [],
        "compute_seconds": 0,
        "processed_samples": 0,
        "complete": False,
    }
    manifest_path = output / "manifest.json"
    checkpoint_path = output / "checkpoint.json"
    if existed:
        manifest = json.loads(manifest_path.read_text())
        if manifest["signature"] != signature:
            cap.release()
            raise ValueError("Input/model/config/code changed; cannot resume")
        state = json.loads(checkpoint_path.read_text())
        if state["complete"]:
            cap.release()
            return json.loads((output / "summary.json").read_text())
        # A shot may be durable before its checkpoint. Discard uncommitted shots.
        for path in (output / "shots").glob("*.jsonl"):
            if int(path.stem) >= state["shot"]:
                path.unlink()
    else:
        manifest = {
            "schema_version": 1,
            "signature": signature,
            "video": str(video),
            "video_end": video_end,
            "fps": fps,
            "runtime": {
                "python": platform.python_version(),
                "torch": version("torch"),
                "ultralytics": version("ultralytics"),
                "platform": platform.platform(),
                "packages": packages,
            },
            "status": "experimental",
            "segmentation": "HSV scene cuts + grass/person wide view + repeated visual content replay candidates; no trained replay classifier",
        }
        atomic_json(manifest_path, manifest)
        atomic_json(checkpoint_path, state)
    start = time.perf_counter()
    owned_mapper = mapper is None
    try:
        detector = detector or Detector(checkpoint, device, config)
        manifest["runtime"]["hardware"] = getattr(detector, "hardware", "test-injected")
        atomic_json(manifest_path, manifest)
        mapper = mapper or PitchMapper(
            pitch_root or Path(__file__).resolve().parents[1] / "artifacts/pnlcalib",
            device,
            config,
        )
    except Exception:
        cap.release()
        raise
    segmenter = Segmenter(config, state["replay_history"])
    team = TeamAssigner(config.team_margin, state["team_state"])
    people_motion = Motion(config.prediction_seconds, 15)
    ball_motion = Motion(config.prediction_seconds, config.max_ball_speed_mps)
    possession = Possession(
        config.possession_persistence,
        config.possession_radius_m,
        config.ambiguity_margin_m,
    )
    identities = {}
    shot = state["shot"]
    samples = state["processed_samples"]
    frame_index = state["next_frame"]
    next_due = state.get("next_due", frame_index / fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    partial = output / "active-shot.partial"
    file = partial.open("w")
    pending = 0

    def commit(next_frame, complete=False, sample_due=None):
        nonlocal shot, file, pending
        file.flush()
        os.fsync(file.fileno())
        file.close()
        if pending:
            partial.replace(output / "shots" / f"{shot:06d}.jsonl")
            shot += 1
        else:
            partial.unlink(missing_ok=True)
        state.update(
            next_frame=next_frame,
            next_due=next_due if sample_due is None else sample_due,
            shot=shot,
            team_state=team.snapshot(),
            replay_history=segmenter.history,
            compute_seconds=state["compute_seconds"] + time.perf_counter() - start,
            processed_samples=samples,
            complete=complete,
        )
        # Compute time is accumulated only at commits, using a separate timer below.
        atomic_json(checkpoint_path, state)
        pending = 0
        if not complete:
            file = partial.open("w")

    last_timestamp = None
    last_decoded = None
    decoded_delta = 1 / fps
    previous_view = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if last_decoded is not None and timestamp > last_decoded:
                decoded_delta = timestamp - last_decoded
            last_decoded = timestamp
            if max_seconds is not None and timestamp >= max_seconds:
                break
            if timestamp < next_due - 1e-8:
                frame_index += 1
                continue
            if last_timestamp is not None and timestamp <= last_timestamp:
                raise ValueError("Decoder timestamps are not strictly increasing")
            last_timestamp = timestamp
            due_before_sample = next_due
            while next_due <= timestamp + 1e-8:
                next_due += 1 / config.sample_hz
            detection = detector.predict(frame)
            player_count = int(np.sum(detection.boxes.cls.cpu().numpy() == 2))
            view = segmenter.update(frame, timestamp, player_count)
            if view["cut"]:
                # History for the new shot frame belongs after the completed shot.
                history = segmenter.history
                segmenter.history = history[:-1]
                commit(frame_index, sample_due=due_before_sample)
                start = time.perf_counter()
                segmenter.history = history
                detector.reset()
                mapper.reset()
                team.reset_shot()
                people_motion.reset()
                ball_motion.reset()
                possession.reset()
                identities = {}
            if not view["eligible_view"] or previous_view is False:
                people_motion.reset()
                ball_motion.reset()
                possession.reset()
            previous_view = view["eligible_view"]
            calibration = (
                mapper.update(frame, timestamp)
                if view["eligible_view"]
                else {"valid": False, "method": "excluded_view", "homography": None}
            )
            if not calibration["valid"]:
                mapper.reset()
                people_motion.reset()
                ball_motion.reset()
                possession.reset()
            tracks = detector.track(detection, frame)
            people = []
            observed = set()
            for track in tracks:
                x1, y1, x2, y2, identity, confidence, klass = track[:7]
                identity = f"{shot}:{int(identity)}"
                klass = int(klass)
                bbox = [float(x1), float(y1), float(x2), float(y2)]
                assigned = (
                    team.observe(identity, colour(frame, bbox))
                    if klass == 2 and view["eligible_view"]
                    else None
                )
                xy = (
                    mapper.project([(x1 + x2) / 2, y2])
                    if calibration["valid"]
                    else None
                )
                person = {
                    "id": identity,
                    "role": NAMES[klass],
                    "team": assigned,
                    "bbox": bbox,
                    "confidence": float(confidence),
                    "xy": xy,
                    "observation": "observed",
                }
                people.append(person)
                observed.add(identity)
                identities[identity] = person
                if xy is not None:
                    people_motion.observe(identity, xy, timestamp)
            if calibration["valid"] and view["eligible_view"]:
                for identity, prior in list(identities.items()):
                    if identity not in observed:
                        xy = people_motion.predict(identity, timestamp)
                        if (
                            xy is not None
                            and 0 <= xy[0] <= config.pitch_length
                            and 0 <= xy[1] <= config.pitch_width
                        ):
                            people.append(
                                {
                                    **prior,
                                    "xy": xy,
                                    "observation": "predicted",
                                    "bbox": None,
                                }
                            )
            balls = detection.boxes[detection.boxes.cls == 0]
            raw_balls = [
                {
                    "role": "ball",
                    "bbox": [float(v) for v in balls.xyxy[i].cpu().numpy()],
                    "confidence": float(balls.conf[i]),
                }
                for i in range(len(balls))
            ]
            ball = {
                "xy": None,
                "pixel": None,
                "observation": "unknown",
                "airborne_uncertain": False,
            }
            if len(balls) and calibration["valid"]:
                order = int(balls.conf.argmax())
                bbox = balls.xyxy[order].cpu().numpy()
                pixel = [float((bbox[0] + bbox[2]) / 2), float((bbox[1] + bbox[3]) / 2)]
                xy = mapper.project(pixel)
                if ball_motion.observe("ball", xy, timestamp):
                    ball.update(
                        xy=xy,
                        pixel=pixel,
                        bbox=[float(v) for v in bbox],
                        confidence=float(balls.conf[order]),
                        observation="observed",
                    )
                    nearby = [
                        p
                        for p in people
                        if p["bbox"]
                        and abs(pixel[0] - (p["bbox"][0] + p["bbox"][2]) / 2)
                        < max(20, p["bbox"][2] - p["bbox"][0])
                    ]
                    ball["airborne_uncertain"] = bool(
                        nearby
                        and all(
                            pixel[1]
                            < p["bbox"][3] - 0.3 * (p["bbox"][3] - p["bbox"][1])
                            for p in nearby
                        )
                    )
            if ball["xy"] is None and calibration["valid"] and view["eligible_view"]:
                xy = ball_motion.predict("ball", timestamp)
                if (
                    xy is not None
                    and 0 <= xy[0] <= config.pitch_length
                    and 0 <= xy[1] <= config.pitch_width
                ):
                    ball.update(xy=xy, observation="predicted")
            eligible = bool(view["eligible_view"] and calibration["valid"])
            measured_ball = ball["xy"] if ball["observation"] == "observed" else None
            control, event = possession.update(
                timestamp, measured_ball, people, eligible, ball["airborne_uncertain"]
            )
            if event:
                event["shot"] = shot
            record = {
                "frame_index": frame_index,
                "timestamp": timestamp,
                "shot": shot,
                "view": view,
                "calibration": calibration,
                "eligible": eligible,
                "people": people,
                "ball": ball,
                "ball_detections": raw_balls,
                "possession": control,
                "event": event,
            }
            file.write(json.dumps(record, allow_nan=False) + "\n")
            pending += 1
            samples += 1
            if samples % 50 == 0:
                print(
                    json.dumps(
                        {
                            "sampled_frames": samples,
                            "video_seconds": round(timestamp, 2),
                            "shot": shot,
                        }
                    ),
                    flush=True,
                )
            frame_index += 1
        commit(frame_index, complete=False)
        # Final exports are recoverable: mark complete only after successful aggregation.
        video_end = (
            min(last_decoded + decoded_delta, max_seconds)
            if max_seconds is not None and last_decoded is not None
            else last_decoded + decoded_delta
            if last_decoded is not None
            else video_end
        )
        manifest["video_end"] = video_end
        result = summarize(output, config, metadata, video_end)
        state["complete"] = True
        manifest["processed_samples"] = samples
        manifest["compute_seconds"] = state["compute_seconds"]
        manifest["processed_samples_per_compute_second"] = (
            samples / state["compute_seconds"] if state["compute_seconds"] else None
        )
        atomic_json(manifest_path, manifest)
        atomic_json(checkpoint_path, state)
        return result
    finally:
        cap.release()
        if not file.closed:
            file.close()
        if owned_mapper:
            mapper.close()
        if pending == 0:
            partial.unlink(missing_ok=True)
