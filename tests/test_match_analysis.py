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
        for i in range(160):
            team.observe(str(i % 2), [0.8, 0, 0.8] if i % 2 else [-0.8, 0, 0.8])
        a = team.observe("left", [-0.8, 0, 0.8])
        team.observe("left", [-0.8, 0, 0.8])
        a = team.observe("left", [-0.8, 0, 0.8])
        self.assertEqual(a, "A")
        self.assertIsNone(team.observe("unknown", [0, 0, 0.8]))
        restored = TeamAssigner(state=team.snapshot())
        self.assertTrue(np.allclose(team.centers, restored.centers))

    def test_team_initialization_rejects_sparse_colour_outliers(self):
        team = TeamAssigner()
        for i in range(180):
            c = (
                [0.6, -0.3, 0.4]
                if i % 20 == 0
                else ([0, 0.1, 0.7] if i % 2 else [0.4, 0.7, 0.85])
            )
            team.observe(str(i % 20), c)
        self.assertIsNotNone(team.centers)
        self.assertLess(
            np.min(np.linalg.norm(team.centers - np.array([0, 0.1, 0.7]), axis=1)), 0.01
        )
        self.assertLess(
            np.min(np.linalg.norm(team.centers - np.array([0.4, 0.7, 0.85]), axis=1)),
            0.01,
        )

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


class CoachingIterationTests(unittest.TestCase):
    def test_tiling_covers_edges_and_nms_retains_different_classes(self):
        from match_analysis.detection import suppress, tiles

        regions = tiles(1920, 1080)
        self.assertTrue(any(x2 == 1920 and y2 == 1080 for x, y, x2, y2 in regions))
        self.assertTrue(
            all(0 <= x < x2 <= 1920 and 0 <= y < y2 <= 1080 for x, y, x2, y2 in regions)
        )
        rows = suppress(
            [[0, 0, 10, 10, 0.8, 0], [0, 0, 10, 10, 0.7, 0], [0, 0, 10, 10, 0.7, 2]]
        )
        self.assertEqual(len(rows), 2)

    def test_ball_candidate_ambiguity_and_motion_gate(self):
        from match_analysis.state import BallSelector

        motion = Motion()
        selector = BallSelector(motion)

        def ball(x, c=0.8):
            return {"xy": [x, 10], "confidence": c}

        selected, reason = selector.select([ball(10), ball(30, 0.79)], 0)
        self.assertIsNone(selected)
        self.assertEqual(reason, "ambiguous_ball_candidates")
        selected, _ = selector.select([ball(10)], 0)
        self.assertIsNotNone(selected)
        selected, _ = selector.select([ball(10.5, 0.65), ball(90, 0.99)], 0.1)
        self.assertEqual(selected["xy"], [10.5, 10])
        motion.reset()
        selected, _ = selector.select([ball(11, 0.65)], 2)
        self.assertIsNone(selected)

    def test_coaching_abstains_with_missing_predicted_or_duplicated_players(self):
        from match_analysis.coaching import frame_features

        frame = {
            "timestamp": 0,
            "shot": 0,
            "eligible": True,
            "ball": {"observation": "unknown"},
            "possession": "unknown",
            "people": [],
        }
        for i in range(6):
            frame["people"].append(
                {
                    "id": str(i),
                    "role": "player",
                    "team": "A",
                    "xy": [i * 2, i],
                    "observation": "observed",
                    "confidence": 0.8,
                }
            )
        result = frame_features(frame, Config())
        self.assertEqual(result["teams"]["A"]["width_m"], 5)
        frame["people"][0]["observation"] = "predicted"
        self.assertIsNone(frame_features(frame, Config())["teams"]["A"]["width_m"])
        frame["people"].append(frame["people"][-1])
        self.assertEqual(
            frame_features(frame, Config())["teams"]["A"]["observed_outfield_count"], 5
        )
        frame["eligible"] = False
        self.assertEqual(
            frame_features(frame, Config())["teams"]["A"]["reason"], "excluded_interval"
        )

    def test_person_identity_matching_and_role_assessment_are_distinct(self):
        from match_analysis.evaluation import match_boxes

        a = [{"role": "referee", "bbox": [0, 0, 10, 20]}]
        b = [{"role": "player", "bbox": [0, 0, 10, 20]}]
        self.assertEqual(match_boxes(a, b), [])
        self.assertEqual(match_boxes(a, b, role_sensitive=False), [(0, 0)])

    def test_requested_partitions_cannot_split_one_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "sources.json"
            p.write_text(
                json.dumps(
                    [
                        {
                            "clip_id": "one",
                            "match_id": "same",
                            "requested_partition": "development",
                        },
                        {
                            "clip_id": "two",
                            "match_id": "same",
                            "requested_partition": "evaluation",
                        },
                    ]
                )
            )
            with self.assertRaises(ValueError):
                freeze(p, root / "frozen.json")

    def test_review_requires_explicit_coverage_and_preserves_uncertain(self):
        from match_analysis.annotations import reviewed

        truth = {"clip_id": "one", "frames": [{"timestamp": 0}, {"timestamp": 0.1}]}
        with self.assertRaises(ValueError):
            reviewed(truth, [], [], "human", "hash")
        data = reviewed(
            truth, [{"start": 0, "end": 0.2, "view": "uncertain"}], [], "human", "hash"
        )
        self.assertNotIn("eligible", data["frames"][0])
        self.assertEqual(data["review"]["reviewer"], "human")
        with self.assertRaises(ValueError):
            reviewed(
                truth, [{"start": 0, "end": 0.2, "view": "eligible"}], [], "", "hash"
            )


class EvaluationProtocolTests(unittest.TestCase):
    def test_event_labels_are_mapped_and_identity_does_not_require_correct_role(self):
        from match_analysis.evaluation import evaluate
        from match_analysis.hashing import digest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "dummy.mp4"
            video.write_bytes(b"behavioral fixture only")
            gs = [
                {"id": "r", "role": "referee", "bbox": [0, 0, 10, 20]},
                {
                    "id": "a",
                    "role": "player",
                    "team": "left",
                    "bbox": [20, 0, 30, 20],
                    "xy": [1, 1],
                },
                {
                    "id": "b",
                    "role": "player",
                    "team": "right",
                    "bbox": [40, 0, 50, 20],
                    "xy": [10, 1],
                },
            ]
            truth = {
                "clip_id": "fixture",
                "match_id": "fixture_match",
                "team_mapping": {"A": "left", "B": "right"},
                "transitions_use_ground_truth_team_labels": True,
                "frames": [{"frame_index": 0, "people": gs}],
                "transitions": [
                    {"timestamp": 2, "from_team": "left", "to_team": "right"}
                ],
            }
            annotations = root / "gt.json"
            annotations.write_text(json.dumps(truth))
            sources = root / "sources.json"
            sources.write_text(
                json.dumps(
                    [
                        {
                            "clip_id": "fixture",
                            "match_id": "fixture_match",
                            "video": str(video),
                            "annotations": str(annotations),
                        }
                    ]
                )
            )
            partition = root / "frozen.json"
            freeze(sources, partition)
            run = root / "run"
            run.mkdir()
            (run / "shots").mkdir()
            (run / "manifest.json").write_text(
                json.dumps({"signature": {"video_sha256": digest(video)}})
            )
            (run / "checkpoint.json").write_text(json.dumps({"complete": True}))
            ps = [
                {
                    **g,
                    "id": "p" + g["id"],
                    "role": "player" if g["id"] == "r" else g["role"],
                    "team": None if g["id"] == "r" else "A" if g["id"] == "a" else "B",
                    "observation": "observed",
                }
                for g in gs
            ]
            (run / "shots/000000.jsonl").write_text(
                json.dumps(
                    {"frame_index": 0, "people": ps, "ball": {"observation": "unknown"}}
                )
                + "\n"
            )
            (run / "events.json").write_text(
                json.dumps([{"timestamp": 2, "from_team": "A", "to_team": "B"}])
            )
            result = evaluate(run, annotations, partition, root / "result.json")[
                "metrics"
            ]
            self.assertEqual(result["tracking_idf1"], 1)
            self.assertAlmostEqual(result["role_sensitive_tracking_idf1"], 2 / 3)
            self.assertEqual(result["transitions"]["precision"], 1)


class DetectionStateTests(unittest.TestCase):
    def test_tile_class_filter_is_reset_on_every_full_frame(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from match_analysis.detection import predict

        class Data:
            dtype = "float32"
            device = "cpu"

            def __init__(self, rows):
                self.rows = np.asarray(rows, dtype=float).reshape(-1, 6)

            def cpu(self):
                return self

            def numpy(self):
                return self.rows

        class Result:
            def __init__(self, rows):
                self.boxes = SimpleNamespace(data=Data(rows))

            def update(self, boxes):
                self.boxes.data = Data(boxes)

        class StatefulModel:
            classes = None

            def predict(self, source, **kwargs):
                self.classes = kwargs.get("classes", self.classes)
                rows = [[1, 1, 5, 5, 0.9, 0], [5, 5, 12, 20, 0.9, 2]]
                rows = [r for r in rows if self.classes is None or r[5] in self.classes]
                return [
                    Result(rows)
                    for _ in (source if isinstance(source, list) else [source])
                ]

        fake_torch = SimpleNamespace(as_tensor=lambda rows, **kwargs: rows)
        with patch.dict("sys.modules", {"torch": fake_torch}):
            model = StatefulModel()
            for _ in range(2):
                result = predict(model, np.zeros((32, 32, 3)), "cpu", tile_size=32)
                self.assertIn(2, result.boxes.data.numpy()[:, 5])

            class BallModel:
                def predict(self, source, **kwargs):
                    return [Result([[20, 20, 22, 22, 0.99, 0]])]

            mixed = predict(
                StatefulModel(),
                np.zeros((32, 32, 3)),
                "cpu",
                tile_size=0,
                ball_model=BallModel(),
            )
            boxes = mixed.boxes.data.numpy()
            self.assertEqual(sum(boxes[:, 5] == 2), 1)
            self.assertEqual(boxes[boxes[:, 5] == 0, 0].tolist(), [20])


class AutomaticControlTests(unittest.TestCase):
    def test_teammates_do_not_break_control_and_only_opponent_change_emits(self):
        controller = Possession()
        for i in range(11):
            ps = [
                {"id": str(i), "team": "A", "xy": [0, 0], "observation": "observed"},
                {"id": "other", "team": "A", "xy": [0.2, 0], "observation": "observed"},
            ]
            team, event = controller.update(i / 10, [0, 0], ps, True)
        self.assertEqual(team, "A")
        self.assertIsNone(event)
        for i in range(11, 17):
            team, event = controller.update(i / 10, [0, 0], people("B", (0, 0)), True)
        self.assertEqual(team, "B")
        self.assertEqual(event["method"], "observed_stable_team_colour_change")
        self.assertAlmostEqual(event["timestamp"], 1.1)
        self.assertAlmostEqual(event["confirmed_at"], 1.6)

    def test_opponent_or_unknown_contender_abstains_referee_is_not_controller(self):
        c = Possession()
        for rival in ("B", None):
            ps = people("A", (0, 0)) + people(rival, (0.2, 0))
            self.assertEqual(c.update(0, [0, 0], ps, True)[0], "unknown")
            self.assertEqual(c.evidence["reason"], "contested_between_teams")
        ps = people("A", (0.3, 0)) + [{**people(None, (0, 0))[0], "role": "referee"}]
        for i in range(7):
            team, _ = c.update(i / 10, [0, 0], ps, True)
        self.assertEqual(team, "A")
        self.assertEqual(c.update(0.7, [float("nan"), 0], ps, True)[0], "unknown")

    def test_airborne_heuristic_ignores_unrelated_foreground_head(self):
        from match_analysis.vision import airborne_uncertainty

        ps = [{"bbox": [10, 100, 30, 200], "observation": "observed"}]
        self.assertFalse(airborne_uncertainty([20, 30], ps))
        self.assertTrue(airborne_uncertainty([20, 120], ps))
        self.assertFalse(airborne_uncertainty([20, 195], ps))
        ps[0]["observation"] = "predicted"
        self.assertFalse(airborne_uncertainty([20, 120], ps))

    def test_control_reprocessing_preserves_observations_and_exports_real_event_schema(
        self,
    ):
        from match_analysis.control import recompute
        from match_analysis.hashing import digest

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            (source / "shots").mkdir(parents=True)
            (source / "checkpoint.json").write_text(json.dumps({"complete": True}))
            (source / "manifest.json").write_text(
                json.dumps(
                    {
                        "video_end": 3.1,
                        "signature": {"config": Config().as_dict(), "metadata": {}},
                    }
                )
            )
            records = []
            for i in range(31):
                team = "A" if i <= 10 else "B"
                ps = [
                    {
                        "id": team,
                        "role": "player",
                        "team": team,
                        "xy": [0, 0],
                        "bbox": [0, 0, 10, 20],
                        "observation": "observed",
                        "confidence": 0.9,
                    }
                ]
                records.append(
                    {
                        "frame_index": i,
                        "timestamp": i / 10,
                        "shot": 0,
                        "eligible": True,
                        "people": ps,
                        "ball": {"xy": [0, 0], "observation": "observed"},
                        "possession": "unknown",
                        "event": None,
                    }
                )
            shot = source / "shots/000000.jsonl"
            shot.write_text("".join(json.dumps(f) + "\n" for f in records))
            before = digest(shot)
            output = Path(tmp) / "derived"
            result = recompute(source, output)
            self.assertEqual(result["possession_changes"], 1)
            self.assertEqual(digest(shot), before)
            actual = [
                json.loads(s)
                for s in (output / "shots/000000.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [f["people"] for f in actual], [f["people"] for f in records]
            )
            self.assertEqual([f["ball"] for f in actual], [f["ball"] for f in records])
            self.assertIn(
                "observed_stable_team_colour_change",
                (output / "events.csv").read_text(),
            )
            self.assertFalse(
                json.loads((output / "possession_diagnostics.json").read_text())[
                    "runtime_human_input_required"
                ]
            )
            with self.assertRaises(ValueError):
                recompute(source, output)


if __name__ == "__main__":
    unittest.main()
