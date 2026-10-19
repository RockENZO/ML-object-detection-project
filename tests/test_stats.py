import json
import tempfile
import unittest
from pathlib import Path

from match_analysis.stats import player_stats, space_control
from match_analysis.viewer import model_card

HZ = 10.0
IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1]
PITCH = {"length": 105.0, "width": 68.0}


def frame(t, rows, shot=0, h=None):
    return {"t": round(t, 3), "s": shot, "e": 1, "p": rows, "b": None, "c": None, "h": h}


def run(speed, seconds, kind=0, identity=1, start=10.0, y=20.0):
    return [
        frame(i / HZ, [[identity, kind, start + speed * i / HZ, y, 1]])
        for i in range(int(seconds * HZ))
    ]


class PlayerStatsTests(unittest.TestCase):
    def test_constant_speed_gives_distance_and_speed(self):
        rows = player_stats(run(5.0, 10), HZ, 10)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["team"], "A")
        self.assertAlmostEqual(row["distance_m"], 49.5, delta=2.5)
        self.assertAlmostEqual(row["avg_kmh"], 18.0, delta=1.0)
        self.assertAlmostEqual(row["peak_kmh"], 18.0, delta=1.0)
        self.assertEqual(row["high_speed_s"], 0.0)  # 5 m/s is below the 5.5 m/s threshold
        self.assertEqual(row["sprint_s"], 0.0)

    def test_high_speed_counts_only_above_threshold(self):
        row = player_stats(run(6.0, 6), HZ, 6)[0]
        self.assertGreater(row["high_speed_s"], 3.0)
        self.assertEqual(row["sprint_s"], 0.0)
        row = player_stats(run(8.0, 6), HZ, 6)[0]
        self.assertGreater(row["sprint_s"], 3.0)

    def test_identity_jump_is_not_counted_as_distance(self):
        frames = run(2.0, 5) + [
            frame(5 + i / HZ, [[1, 0, 90.0 + 2.0 * i / HZ, 20.0, 1]]) for i in range(50)
        ]
        row = player_stats(frames, HZ, 10)[0]
        self.assertLess(row["distance_m"], 25)
        self.assertLess(row["peak_kmh"], 12 * 3.6)

    def test_camera_cut_splits_segments(self):
        frames = run(2.0, 4) + [
            frame(4 + i / HZ, [[1, 0, 80.0 + 2.0 * i / HZ, 20.0, 1]], shot=1) for i in range(40)
        ]
        row = player_stats(frames, HZ, 8)[0]
        self.assertAlmostEqual(row["distance_m"], 16.0, delta=2.0)

    def test_short_tracks_unobserved_and_referees_are_skipped(self):
        self.assertEqual(player_stats(run(2.0, 1), HZ, 5), [])
        self.assertEqual(player_stats(run(2.0, 5, kind=4), HZ, 5), [])
        predicted = [frame(i / HZ, [[1, 0, 10.0, 20.0, 0]]) for i in range(50)]
        self.assertEqual(player_stats(predicted, HZ, 5), [])

    def test_goalkeeper_has_no_team(self):
        row = player_stats(run(1.0, 5, kind=3), HZ, 5)[0]
        self.assertIsNone(row["team"])
        self.assertEqual(row["role"], "goalkeeper")

    def test_unassigned_samples_still_count_and_the_team_comes_from_the_assigned_ones(self):
        mixed = [frame(i / HZ, [[1, 2 if i < 30 else 0, 10.0 + 2.0 * i / HZ, 20.0, 1]]) for i in range(60)]
        row = player_stats(mixed, HZ, 6)[0]
        self.assertEqual(row["team"], "A")
        self.assertAlmostEqual(row["seconds"], 6.0, delta=0.3)
        self.assertAlmostEqual(row["distance_m"], 12.0, delta=1.0)
        never = player_stats(run(2.0, 5, kind=2), HZ, 5)
        self.assertEqual(len(never), 1)
        self.assertIsNone(never[0]["team"])
        self.assertEqual(never[0]["role"], "player")

    def test_a_split_team_is_not_guessed(self):
        frames = [frame(i / HZ, [[1, 0 if i % 2 else 1, 10.0 + i / HZ, 20.0, 1]]) for i in range(40)]
        self.assertIsNone(player_stats(frames, HZ, 4)[0]["team"])

    def test_a_jump_is_cut_before_smoothing_not_spread_over_steps(self):
        steady = [frame(i / HZ, [[1, 0, 10.0 + 1.0 * i / HZ, 20.0, 1]]) for i in range(40)]
        jumped = [frame((40 + i) / HZ, [[1, 0, 50.0 + 1.0 * i / HZ, 20.0, 1]]) for i in range(40)]
        row = player_stats(steady + jumped, HZ, 8)[0]
        self.assertAlmostEqual(row["distance_m"], 8.0, delta=1.0)
        self.assertLess(row["peak_kmh"], 6.0)

    def test_smoothing_follows_the_sample_rate(self):
        for hz in (2.0, 5.0, 30.0):
            frames = [
                frame(i / hz, [[1, 0, 10.0 + 5.0 * i / hz, 20.0, 1]]) for i in range(int(8 * hz))
            ]
            row = player_stats(frames, hz, 8)[0]
            self.assertAlmostEqual(row["avg_kmh"], 18.0, delta=1.5, msg=str(hz))
            self.assertAlmostEqual(row["distance_m"], 40.0, delta=3.0, msg=str(hz))

    def test_peak_is_unavailable_rather_than_zero_without_a_full_window(self):
        frames = []
        for burst in range(5):
            start = burst * 1.0
            frames += [
                frame(start + i / HZ, [[1, 0, 10.0 + 3.0 * (start + i / HZ), 20.0, 1]]) for i in range(6)
            ]
        row = player_stats(frames, HZ, 6)[0]
        self.assertIsNone(row["peak_kmh"])
        self.assertGreater(row["distance_m"], 5)
        self.assertGreater(row["avg_kmh"], 5)

    def test_output_is_json_serialisable(self):
        json.dumps(player_stats(run(3.0, 5), HZ, 5), allow_nan=False)


class SpaceControlTests(unittest.TestCase):
    size = (105, 68)

    def frames(self, a_x, b_x):
        rows = [[i, 0, a_x, 10.0 + 12 * i, 1] for i in range(3)]
        rows += [[10 + i, 1, b_x, 10.0 + 12 * i, 1] for i in range(3)]
        return [frame(i / HZ, rows, h=IDENTITY) for i in range(20)]

    def test_symmetric_teams_split_the_view_evenly(self):
        result = space_control(self.frames(30.0, 70.0), PITCH, self.size, 2)
        self.assertAlmostEqual(result["mean_A"], (50 - 0) / 105, delta=0.06)
        self.assertEqual(result["supported_fraction"], 1.0)

    def test_team_further_forward_controls_more(self):
        even = space_control(self.frames(30.0, 70.0), PITCH, self.size, 2)["mean_A"]
        ahead = space_control(self.frames(50.0, 70.0), PITCH, self.size, 2)["mean_A"]
        self.assertGreater(ahead, even)

    def test_too_few_players_or_no_homography_is_withheld(self):
        few = [frame(i / HZ, [[1, 0, 30.0, 20.0, 1], [2, 1, 60.0, 20.0, 1]], h=IDENTITY) for i in range(10)]
        self.assertIsNone(space_control(few, PITCH, self.size, 1)["series"][0]["A"])
        none = self.frames(30.0, 70.0)
        for f in none:
            f["h"] = None
        self.assertEqual(space_control(none, PITCH, self.size, 2)["supported_fraction"], 0.0)

    def test_coverage_and_the_mean_are_weighted_by_supported_sample_time(self):
        # One qualifying frame per second at 10 Hz is a tenth of the time, not all of it.
        sparse = [f for i, f in enumerate(self.frames(30.0, 70.0)) if i % 10 == 0]
        for index, f in enumerate(sparse):
            f["t"] = float(index)
        result = space_control(sparse, PITCH, self.size, 2)
        self.assertAlmostEqual(result["supported_fraction"], 0.1, places=2)
        self.assertAlmostEqual(result["series"][0]["support"], 0.1, places=2)
        full = space_control(self.frames(30.0, 70.0), PITCH, self.size, 2)
        self.assertEqual(full["series"][0]["support"], 1.0)

    def test_only_the_camera_visible_area_counts(self):
        # Identity homography: the image is 105 x 68 pixels, so the whole pitch is visible.
        full = space_control(self.frames(30.0, 70.0), PITCH, self.size, 2)["series"][0]["area_m2"]
        half = space_control(self.frames(30.0, 70.0), PITCH, (52, 68), 2)["series"][0]["area_m2"]
        self.assertAlmostEqual(full, 105 * 68, delta=250)
        self.assertAlmostEqual(half, 52 * 68, delta=250)


class ModelCardTests(unittest.TestCase):
    def report(self, **extra):
        data = {
            "checkpoint_sha256": "ab" * 32,
            "options": {"imgsz": 960},
            "validation": {
                "map50": 0.8641,
                "map50_95": 0.641,
                "precision": 0.927,
                "recall": 0.815,
                "per_class": [
                    {"Class": "ball", "Instances": 112, "Box-P": 1.0, "Box-R": 0.432, "mAP50": 0.586}
                ],
            },
        }
        data.update(extra)
        return data

    def write(self, data):
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        self.addCleanup(Path(handle.name).unlink)
        json.dump(data, handle)
        handle.close()
        return handle.name

    def test_card_reports_validation_figures(self):
        card = model_card(self.write(self.report()), "ab" * 32)
        self.assertEqual(card["image_size"], 960)
        self.assertEqual(card["map50"], 0.864)
        self.assertEqual(card["classes"][0]["recall"], 0.432)
        self.assertIn("validation", card["split"])

    def test_adapted_ball_runs_do_not_claim_the_detectors_ball_figures(self):
        data = self.report()
        data["validation"]["per_class"].append(
            {"Class": "player", "Instances": 9, "Box-P": 0.9, "Box-R": 0.9, "mAP50": 0.9}
        )
        card = model_card(self.write(data), "ab" * 32, ball_from_other_model=True)
        self.assertEqual([c["name"] for c in card["classes"]], ["player"])

    def test_card_for_another_checkpoint_is_refused(self):
        with self.assertRaises(ValueError):
            model_card(self.write(self.report()), "cd" * 32)


if __name__ == "__main__":
    unittest.main()
