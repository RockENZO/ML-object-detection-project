import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from match_analysis.config import Config
from match_analysis.evaluation import freeze, transition_metrics
from match_analysis.state import Motion, Possession, TeamAssigner
from match_analysis.summary import summarize, validate_metadata
from match_analysis.vision import Segmenter


def people(team="A", xy=(1, 1), observation="observed"):
    return [{"team": team, "xy": xy, "observation": observation}]


class StateTests(unittest.TestCase):
    def test_prediction_expires_and_reset_stops_it(self):
        motion = Motion()
        motion.observe("x", [1, 1], 0)
        motion.observe("x", [2, 1], 0.1)
        self.assertIsNotNone(motion.predict("x", 1.1))
        self.assertIsNone(motion.predict("x", 1.1001))
        motion.reset()
        self.assertIsNone(motion.predict("x", 0.2))

    def test_transition_first_supported_time(self):
        p = Possession()
        for t in np.arange(0, 0.61, 0.1):
            p.update(float(t), (1, 1), people(), True)
        event = None
        for t in np.arange(0.7, 1.21, 0.1):
            _, event = p.update(float(t), (1, 1), people("B"), True)
        self.assertAlmostEqual(event["timestamp"], 0.7)
        self.assertEqual(event["from_team"], "A")
        self.assertAlmostEqual(event["uncertainty"][0], 0.6)

    def test_missing_airborne_predicted_ambiguous_no_events(self):
        for mode in (
            "missing",
            "airborne",
            "predicted",
            "ambiguous",
            "unassigned",
            "excluded",
        ):
            p = Possession()
            for t in np.arange(0, 0.61, 0.1):
                p.update(float(t), (1, 1), people(), True)
            ball = None if mode == "missing" else (1, 1)
            ps = people(observation="predicted") if mode == "predicted" else people()
            if mode == "ambiguous":
                ps += people("B")
            if mode == "unassigned":
                ps = people(None) + people("A", (2, 1))
            control, event = p.update(
                0.7, ball, ps, mode != "excluded", mode == "airborne"
            )
            self.assertEqual(control, "unknown", mode)
            for t in np.arange(0.8, 1.41, 0.1):
                _, event = p.update(float(t), (1, 1), people("B"), True)
            self.assertIsNone(event, mode)

    def test_team_cluster_persists_and_ambiguous_abstains(self):
        team = TeamAssigner()
        for i in range(80):
            team.observe(str(i % 2), [0.8, 0, 0.8] if i % 2 else [-0.8, 0, 0.8])
        a = team.observe("left", [-0.8, 0, 0.8])
        team.observe("left", [-0.8, 0, 0.8])
        a = team.observe("left", [-0.8, 0, 0.8])
        self.assertEqual(a, "A")
        self.assertIsNone(team.observe("unknown", [0, 0, 0.8]))
        restored = TeamAssigner(state=team.snapshot())
        self.assertTrue(np.allclose(team.centers, restored.centers))

    def test_cut_and_repeated_sequence_excluded(self):
        config = Config()
        segment = Segmenter(config)
        green = np.zeros((100, 100, 3), np.uint8)
        green[:] = (0, 180, 0)
        self.assertTrue(segment.update(green, 0, 10)["eligible_view"])
        white = np.full_like(green, 255)
        self.assertTrue(segment.update(white, 0.1, 0)["cut"])
        self.assertFalse(segment.update(white, 0.2, 0)["eligible_view"])
        segment = Segmenter(config)
        for t in np.arange(0, 1, 0.1):
            segment.update(green, float(t), 10)
        for t in np.arange(12, 13, 0.1):
            value = segment.update(green, float(t), 10)
        self.assertEqual(value["view"], "replay_candidate")
        self.assertFalse(value["eligible_view"])

    def test_direction_halftime_and_invalid_intervals(self):
        metadata = validate_metadata(
            {
                "attacking_directions": [
                    {"start": 0, "end": 1, "A": 1, "B": -1},
                    {"start": 1, "end": 2, "A": -1, "B": 1},
                ]
            }
        )
        from match_analysis.summary import direction_at

        self.assertEqual(direction_at(metadata, 0.9, "A"), 1)
        self.assertEqual(direction_at(metadata, 1, "A"), -1)
        with self.assertRaises(ValueError):
            validate_metadata(
                {"attacking_directions": [{"start": 0, "end": 2, "A": 1, "B": 1}]}
            )

    def test_unique_transition_matching_and_no_unmeasured_pass(self):
        a = {"timestamp": 2, "from_team": "A", "to_team": "B"}
        result = transition_metrics(
            [a], [{**a, "timestamp": 2.9}, {**a, "timestamp": 3}]
        )
        self.assertEqual(result["true_positives"], 1)
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(transition_metrics([a], [])["recall"], 0)

    def test_summary_predictions_excluded_and_unknown_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "shots").mkdir()
            records = []
            for i in range(10):
                records.append(
                    {
                        "timestamp": i / 10,
                        "shot": 0,
                        "eligible": True,
                        "possession": "A" if i < 5 else "unknown",
                        "ball": {"xy": [75, 34]},
                        "people": [
                            {
                                "id": "0:1",
                                "role": "player",
                                "team": "A",
                                "xy": [75, 34],
                                "observation": "predicted",
                            }
                        ],
                    }
                )
            (root / "shots/000000.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in records)
            )
            result = summarize(root, Config(), {}, 1)
            self.assertAlmostEqual(result["possession_coverage"], 0.5)
            self.assertAlmostEqual(result["unknown_eligible_seconds"], 0.5)
            self.assertTrue(
                all(
                    r["A_territorial_pressure"] is None
                    for r in json.loads((root / "trends.json").read_text())
                )
            )




class Array(np.ndarray):
    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


class Boxes:
    def __init__(self, classes=()):
        self.cls = np.asarray(classes, dtype=float).view(Array)

    def __getitem__(self, key):
        return Boxes(self.cls[key])

    def __len__(self):
        return len(self.cls)


class FakeDetector:
    def __init__(self, interrupt=None):
        self.calls = 0
        self.interrupt = interrupt

    def reset(self):
        pass

    def predict(self, frame):
        self.calls += 1
        if self.calls == self.interrupt:
            raise RuntimeError("Simulated interruption")
        from types import SimpleNamespace

        return SimpleNamespace(boxes=Boxes([2] * 6))

    def track(self, detection, frame):
        return []


class FakeMapper:
    def reset(self):
        pass

    def update(self, frame, t):
        return {"valid": False, "method": "unknown", "homography": None}

    def project(self, p):
        return None


class RecoveryTests(unittest.TestCase):
    def test_interrupted_shot_reprocessed_resume_matches_clean(self):
        import cv2

        from match_analysis.pipeline import analyze

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "video.mp4"
            model = root / "model.pt"
            model.write_bytes(b"model")
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64)
            )
            for i in range(30):
                writer.write(
                    np.full(
                        (64, 64, 3),
                        (0, 180, 0) if i < 10 else (255, 255, 255),
                        np.uint8,
                    )
                )
            writer.release()
            with self.assertRaises(RuntimeError):
                analyze(
                    video,
                    model,
                    root / "interrupted",
                    detector=FakeDetector(16),
                    mapper=FakeMapper(),
                )
            checkpoint = json.loads((root / "interrupted/checkpoint.json").read_text())
            self.assertEqual(checkpoint["shot"], 1)
            resumed = analyze(
                video,
                model,
                root / "interrupted",
                resume=True,
                detector=FakeDetector(),
                mapper=FakeMapper(),
            )
            clean = analyze(
                video,
                model,
                root / "clean",
                detector=FakeDetector(),
                mapper=FakeMapper(),
            )
            self.assertEqual(resumed, clean)
            self.assertEqual(resumed["sampled_frames"], 30)
            self.assertEqual(
                list((root / "interrupted/shots").glob("*.jsonl")).__len__(), 2
            )
            self.assertEqual(
                analyze(
                    video,
                    model,
                    root / "interrupted",
                    resume=True,
                    detector=FakeDetector(),
                    mapper=FakeMapper(),
                ),
                resumed,
            )
            model.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                analyze(
                    video,
                    model,
                    root / "interrupted",
                    resume=True,
                    detector=FakeDetector(),
                    mapper=FakeMapper(),
                )

    def test_video_range_seeking_and_unregistered_paths(self):
        import urllib.error
        import urllib.request

        from match_analysis.media import VideoServer

        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "x.mp4"
            video.write_bytes(bytes(range(100)))
            server = VideoServer()
            try:
                url = server.register(video)
                response = urllib.request.urlopen(
                    urllib.request.Request(url, headers={"Range": "bytes=10-19"})
                )
                self.assertEqual(response.status, 206)
                self.assertEqual(response.read(), bytes(range(10, 20)))
                with self.assertRaises(urllib.error.HTTPError):
                    urllib.request.urlopen(
                        url.replace(url.split("/")[-1], "../../etc/passwd")
                    )
            finally:
                server.close()


class CalibrationTests(unittest.TestCase):
    def test_centred_model_coordinates_are_shifted_before_bounds_check(self):
        import io
        from types import SimpleNamespace

        from match_analysis.calibration import PitchMapper

        mapper = PitchMapper.__new__(PitchMapper)
        mapper.config = Config()
        mapper.reset()
        mapper.process = SimpleNamespace(stdin=io.StringIO())
        mapper._receive = lambda: {
            "valid": True,
            "error_px": 0.1,
            "ground_landmarks": 4,
            "homography": np.eye(3).tolist(),
        }
        result = mapper.update(np.zeros((100, 100, 3), np.uint8), 0)
        # Identity H returns native x=0,y=0 (midfield), not the corner.
        self.assertTrue(result["valid"])
        self.assertEqual(mapper.project([0, 0]), [52.5, 34.0])
        self.assertEqual(mapper.project([-52.5, -34]), [0.0, 0.0])
        self.assertIsNone(mapper.project([-53, -34]))
        mapper._receive = lambda: {"valid": False}
        self.assertFalse(mapper.update(np.zeros((100, 100, 3), np.uint8), 0.5)["valid"])
        self.assertIsNone(mapper.project([0, 0]))


class PlatformTests(unittest.TestCase):
    def test_cuda_selector_is_portable_between_ultralytics_and_torch(self):
        from match_analysis.pnl_worker import device_string

        self.assertEqual(device_string("0"), "cuda:0")
        self.assertEqual(device_string("mps"), "mps")

    def test_freeze_groups_and_rejects_overwrites_and_mutated_annotations(self):
        from match_analysis.evaluation import evaluate
        from match_analysis.hashing import digest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "v.mp4"
            video.write_bytes(b"video")
            annotation = root / "a.json"
            annotation.write_text(
                json.dumps({"clip_id": "one", "match_id": "match", "frames": []})
            )
            sources = root / "sources.json"
            sources.write_text(
                json.dumps(
                    [
                        {
                            "clip_id": "one",
                            "match_id": "match",
                            "video": str(video),
                            "annotations": str(annotation),
                        },
                        {"clip_id": "two", "match_id": "match"},
                    ]
                )
            )
            frozen = root / "frozen.json"
            result = freeze(sources, frozen)
            self.assertEqual(
                result["sources"][0]["partition"], result["sources"][1]["partition"]
            )
            with self.assertRaises(ValueError):
                freeze(sources, frozen)
            run = root / "run"
            run.mkdir()
            (run / "manifest.json").write_text(
                json.dumps({"signature": {"video_sha256": digest(video)}})
            )
            annotation.write_text(annotation.read_text() + " ")
            with self.assertRaises(ValueError):
                evaluate(run, annotation, frozen, root / "eval.json")

if __name__ == "__main__":
    unittest.main()
