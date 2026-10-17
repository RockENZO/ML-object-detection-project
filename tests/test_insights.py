import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from match_analysis.config import Config
from match_analysis.insights import Settings, build, camera_focus, in_box
from match_analysis.viewer import export_viewer, frame_payload, team_colours

SIZE = [1920, 1080]


def homography(config):
    """Maps the image to the pitch so the image centre lands at the pitch centre."""
    sx, sy = config.pitch_length / SIZE[0], config.pitch_width / SIZE[1]
    return [[sx, 0, 0], [0, sy, 0], [0, 0, 1]]


def person(identity, role, team, xy, observation="observed", confidence=0.9):
    return {
        "id": f"0:{identity}",
        "role": role,
        "team": team,
        "bbox": [10, 10, 50, 120] if observation == "observed" else None,
        "confidence": confidence,
        "xy": list(xy),
        "observation": observation,
    }


def make_frame(config, t, ball_xy, team_a, team_b, possession="unknown", eligible=True):
    people = [person(i, "player", "A", xy) for i, xy in enumerate(team_a)]
    people += [person(100 + i, "player", "B", xy) for i, xy in enumerate(team_b)]
    return {
        "frame_index": int(t * 25),
        "timestamp": t,
        "shot": 0,
        "view": {},
        "calibration": {"valid": eligible, "homography": homography(config) if eligible else None},
        "eligible": eligible,
        "people": people if eligible else [],
        "ball": {
            "xy": list(ball_xy) if ball_xy else None,
            "pixel": [100, 100] if ball_xy else None,
            "observation": "observed" if ball_xy else "unknown",
            "airborne_uncertain": False,
        },
        "ball_detections": [],
        "possession": possession,
        "control_evidence": {},
        "event": None,
    }


def squad(cx, cy, n=8, spread=(9, 12)):
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return [
        (cx + spread[0] * math.cos(a), cy + spread[1] * math.sin(a)) for a in angles
    ]


def write_run(root, config, frames, events=(), video=None):
    (root / "shots").mkdir(parents=True)
    (root / "shots" / "000000.jsonl").write_text(
        "\n".join(json.dumps(f) for f in frames) + "\n"
    )
    signature = {
        "config": config.as_dict(),
        "metadata": {},
        "video_sha256": "0" * 64,
        "model_sha256": "1" * 64,
    }
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "signature": signature,
                "video": str(video or root / "missing.mp4"),
                "video_end": frames[-1]["timestamp"] + 0.1,
                "runtime": {"hardware": "test"},
            }
        )
    )
    (root / "checkpoint.json").write_text(
        json.dumps({"team_state": {"centers": [[0.8, 0.1, 0.9], [-0.2, -0.6, 0.5]]}})
    )
    (root / "events.json").write_text(json.dumps(list(events)))
    (root / "summary.json").write_text(
        json.dumps(
            {
                "durations": {},
                "possession_coverage": 0.5,
                "eligible_fraction": 1.0,
                "observed_possession_percent": {"A": 100.0, "B": 0.0},
                "possession_changes": len(list(events)),
                "team_names": {"A": "Team A", "B": "Team B"},
            }
        )
    )


def attacking_scenario(config, seconds=24):
    """Team A keep the ball and press into the right penalty area; B defend."""
    frames = []
    for i in range(seconds * 10):
        t = i / 10
        progress = min(1.0, t / 14)
        ball_x = 55 + progress * 40
        ball = (ball_x, 34)
        team_a = squad(ball_x - 3, 34, spread=(8, 10))
        team_b = squad(ball_x + 10, 34, n=7, spread=(8, 10))
        frames.append(make_frame(config, t, ball, team_a, team_b, "A"))
    return frames


class InsightTests(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name) / "run"
        self.addCleanup(self.directory.cleanup)

    def run_build(self, frames, events=()):
        write_run(self.root, self.config, frames, events)
        return build(self.root, self.config, {}, list(events), SIZE)

    def test_dominant_team_has_positive_momentum_and_box_moment(self):
        result = self.run_build(attacking_scenario(self.config))
        supported = [m["value"] for m in result["momentum"] if m["value"] is not None]
        self.assertGreater(len(supported), 15)
        self.assertGreater(float(np.mean(supported)), 0.3)
        self.assertTrue(all(-1 <= v <= 1 for v in supported))
        box = [m for m in result["moments"] if m["type"] == "box_entry"]
        self.assertEqual(len(box), 1)
        self.assertEqual(box[0]["team"], "A")
        self.assertIn("right penalty area", box[0]["title"])
        self.assertEqual(box[0]["confidence"], "medium")

    def test_missing_ball_uses_camera_focus_and_says_so(self):
        frames = attacking_scenario(self.config)
        for frame in frames:
            frame["ball"] = {"xy": None, "pixel": None, "observation": "unknown", "airborne_uncertain": False}
            frame["possession"] = "unknown"
        result = self.run_build(frames)
        self.assertEqual(result["coverage"]["ball_observed_fraction"], 0)
        self.assertGreater(result["coverage"]["camera_focus_fraction"], 0.99)
        kinds = {m.get("anchor_kind") for m in result["momentum"]}
        self.assertEqual(kinds, {"camera_focus"})

    def test_no_calibration_means_no_momentum_and_no_moments(self):
        frames = [
            make_frame(self.config, i / 10, None, [], [], eligible=False)
            for i in range(120)
        ]
        result = self.run_build(frames)
        self.assertTrue(all(m["value"] is None for m in result["momentum"]))
        self.assertEqual(result["moments"], [])
        self.assertEqual(result["tactics"], [])
        self.assertEqual(result["coverage"]["eligible_fraction"], 0)

    def test_predicted_players_never_count(self):
        frames = attacking_scenario(self.config)
        for frame in frames:
            for p in frame["people"]:
                p["observation"] = "predicted"
                p["bbox"] = None
        result = self.run_build(frames)
        self.assertEqual(result["coverage"]["mean_observed_players"], 0)
        self.assertEqual(result["tactics"], [])
        self.assertTrue(all("near_anchor" not in m for m in result["momentum"]))

    def test_possession_change_event_becomes_a_moment(self):
        event = {
            "type": "possession_change",
            "from_team": "A",
            "to_team": "B",
            "timestamp": 8.0,
            "uncertainty": [7.5, 8.0],
            "confirmed_at": 8.5,
            "observed_support_seconds": 0.5,
            "support_track_ids": ["0:100"],
        }
        result = self.run_build(attacking_scenario(self.config), [event])
        change = [m for m in result["moments"] if m["type"] == "possession_change"]
        self.assertEqual(len(change), 1)
        self.assertEqual(change[0]["team"], "B")
        self.assertEqual((change[0]["start"], change[0]["end"]), (7.5, 8.5))

    def test_swing_requires_opposite_sustained_runs(self):
        frames = []
        for i in range(300):
            t = i / 10
            first_half = t < 14
            ball = (88, 34) if first_half else (17, 34)
            attackers = squad(85 if first_half else 20, 34, spread=(8, 10))
            defenders = squad(98 if first_half else 7, 34, n=6, spread=(8, 10))
            a, b = (attackers, defenders) if first_half else (defenders, attackers)
            frames.append(make_frame(self.config, t, ball, a, b, "A" if first_half else "B"))
        result = self.run_build(frames)
        swings = [m for m in result["moments"] if m["type"] == "momentum_swing"]
        self.assertEqual(len(swings), 1)
        self.assertEqual(swings[0]["team"], "B")
        self.assertTrue(12 <= swings[0]["t"] <= 18)

    def test_tactic_notes_withheld_below_six_observed_players(self):
        frames = [
            make_frame(self.config, i / 10, (50, 34), squad(45, 34, n=5), squad(60, 34, n=5), "A")
            for i in range(200)
        ]
        result = self.run_build(frames)
        self.assertEqual([n for n in result["tactics"] if n["kind"] in ("compact", "stretched")], [])

    def test_tactic_notes_carry_evidence_and_player_support(self):
        frames = []
        for i in range(300):
            t = i / 10
            wide = t >= 15
            a = squad(40, 34, n=8, spread=(22, 26) if wide else (8, 10))
            b = squad(60, 34, n=8, spread=(9, 12))
            frames.append(make_frame(self.config, t, (50, 34), a, b, "A"))
        result = self.run_build(frames)
        notes = {n["kind"]: n for n in result["tactics"] if n["team"] == "A"}
        self.assertIn("compact", notes)
        self.assertIn("stretched", notes)
        self.assertLess(notes["compact"]["start"], notes["stretched"]["start"])
        for note in result["tactics"]:
            self.assertGreaterEqual(note["players_observed"], 1)
            self.assertTrue(note["evidence"])

    def test_moments_link_overlapping_tactic_notes(self):
        frames = []
        for i in range(300):
            t = i / 10
            wide = t >= 15
            a = squad(88, 34, n=8, spread=(22, 26) if wide else (8, 10))
            b = squad(98, 34, n=8, spread=(9, 12))
            frames.append(make_frame(self.config, t, (95, 34), a, b, "A"))
        result = self.run_build(frames)
        ids = {n["id"] for n in result["tactics"]}
        self.assertTrue(result["moments"])
        for moment in result["moments"]:
            self.assertLessEqual(len(moment["notes"]), 3)
            self.assertTrue(set(moment["notes"]) <= ids)
        self.assertTrue(any(m["notes"] for m in result["moments"]))

    def test_isolated_ball_detection_falls_back_to_camera_focus(self):
        frames = [
            make_frame(self.config, i / 10, (5, 5), squad(60, 34), squad(70, 34, n=7), "unknown")
            for i in range(120)
        ]
        result = self.run_build(frames)
        self.assertEqual(result["coverage"]["ball_observed_fraction"], 1)
        self.assertEqual(result["coverage"]["ball_usable_fraction"], 0)
        self.assertEqual({m.get("anchor_kind") for m in result["momentum"]}, {"camera_focus"})

    def test_tempo_surge_needs_a_clear_rise_over_the_clip_median(self):
        frames = []
        for i in range(300):
            t = i / 10
            run = max(0.0, t - 15) * 5.0  # 5 m/s after 15 s, standing before
            a = [(x + run, y) for x, y in squad(40, 34)]
            b = [(x + run, y) for x, y in squad(55, 34, n=7)]
            frames.append(make_frame(self.config, t, (50 + run, 34), a, b, "A"))
        surge = [m for m in self.run_build(frames)["moments"] if m["type"] == "tempo_surge"]
        self.assertEqual(len(surge), 1)
        self.assertGreaterEqual(surge[0]["start"], 15)
        self.assertEqual(surge[0]["confidence"], "low")

    def test_geometry_helpers(self):
        self.assertEqual(in_box([5, 34], self.config), "left")
        self.assertEqual(in_box([100, 34], self.config), "right")
        self.assertIsNone(in_box([5, 5], self.config))
        self.assertIsNone(in_box([52.5, 34], self.config))
        frame = make_frame(self.config, 0, None, [], [])
        self.assertEqual(camera_focus(frame, SIZE, self.config), [52.5, 34.0])
        self.assertIsNone(camera_focus(frame, None, self.config))
        frame["calibration"]["homography"] = [[0, 0, 500], [0, 0, 500], [0, 0, 1]]
        self.assertIsNone(camera_focus(frame, SIZE, self.config))

    def test_settings_do_not_mutate_between_calls(self):
        a, b = Settings(), Settings()
        a.weights["control"] = 9
        self.assertNotEqual(b.weights["control"], 9)


class ViewerExportTests(unittest.TestCase):
    def test_export_writes_complete_schema_and_copied_video(self):
        import cv2

        config = Config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.mp4"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, tuple(SIZE)
            )
            for _ in range(5):
                writer.write(np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8))
            writer.release()
            frames = attacking_scenario(config, seconds=12)
            write_run(root / "run", config, frames, video=video)
            result = export_viewer(
                root / "run", root / "web", video=video, transcode_video=False
            )
            payload = json.loads((root / "web" / "analysis.json").read_text())
            self.assertTrue((root / "web" / "clip.mp4").is_file())
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["status"], "experimental")
            self.assertEqual(payload["pitch"], {"length": 105.0, "width": 68.0})
            self.assertEqual(len(payload["frames"]), len(frames))
            self.assertEqual(set(payload["teams"]), {"A", "B"})
            self.assertRegex(payload["teams"]["A"]["colour"], r"^#[0-9a-f]{6}$")
            self.assertTrue(payload["limitations"])
            first = payload["frames"][0]
            self.assertEqual(set(first), {"t", "s", "e", "p", "b", "c", "h"})
            self.assertEqual(len(first["h"]), 9)
            self.assertEqual(len(first["p"][0]), 9)
            self.assertEqual(first["b"][2], 1)
            self.assertEqual(result["moments"], len(payload["moments"]))
            json.dumps(payload, allow_nan=False)

    def test_predicted_people_have_no_pixels_and_unassigned_kinds(self):
        config = Config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            frame = make_frame(config, 0, None, [], [])
            frame["people"] = [
                person(1, "player", None, (10, 10)),
                person(2, "goalkeeper", None, (5, 34)),
                person(3, "referee", None, (50, 30)),
                person(4, "player", "B", (60, 30), observation="predicted"),
            ]
            write_run(root, config, [frame])
            rows = frame_payload(root)[0]["p"]
            self.assertEqual([r[1] for r in rows], [2, 3, 4, 1])
            self.assertEqual(len(rows[3]), 5)
            self.assertEqual(rows[3][4], 0)

    def test_team_colours_fall_back_and_stay_legible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoint.json").write_text(json.dumps({"team_state": None}))
            self.assertEqual(set(team_colours(root)), {"A", "B"})
            (root / "checkpoint.json").write_text(
                json.dumps({"team_state": {"centers": [[0, 0, 0.0], [0.0, 0.0, 1.0]]}})
            )
            dark = team_colours(root)["A"]
            self.assertGreaterEqual(int(dark[1:3], 16), 0x9e)


if __name__ == "__main__":
    unittest.main()
