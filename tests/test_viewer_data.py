"""Invariants for the committed static viewer; also run before every Pages deploy."""

import json
import re
import unittest
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"
DATA = WEB / "data"
MOMENT_TYPES = {
    "momentum_swing",
    "sustained_pressure",
    "possession_change",
    "box_entry",
    "fast_ball",
    "crowding",
    "tempo_surge",
}


class ViewerDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads((DATA / "analysis.json").read_text())

    def test_schema_and_status(self):
        p = self.payload
        self.assertEqual(p["schema_version"], 1)
        self.assertEqual(p["status"], "experimental")
        self.assertGreater(p["source"]["duration"], 1)
        self.assertEqual(len(p["source"]["image_size"]), 2)
        self.assertEqual(set(p["teams"]), {"A", "B"})
        for team in p["teams"].values():
            self.assertRegex(team["colour"], r"^#[0-9a-f]{6}$")
        self.assertTrue(p["limitations"])

    def test_every_referenced_video_exists(self):
        for video in self.payload["source"]["videos"]:
            self.assertTrue((DATA / video["src"]).is_file(), video["src"])
            self.assertTrue(video["type"].startswith("video/"))

    def test_frames_are_ordered_and_inside_the_pitch(self):
        pitch = self.payload["pitch"]
        frames = self.payload["frames"]
        self.assertGreater(len(frames), 10)
        times = [f["t"] for f in frames]
        self.assertEqual(times, sorted(times))
        self.assertLessEqual(times[-1], self.payload["source"]["duration"] + 0.5)
        for frame in frames:
            for row in frame["p"]:
                self.assertIn(row[1], range(5))
                self.assertTrue(0 <= row[2] <= pitch["length"], row)
                self.assertTrue(0 <= row[3] <= pitch["width"], row)
                self.assertIn(row[4], (0, 1))
                self.assertIn(len(row), (5, 9))
            if frame["b"]:
                self.assertTrue(0 <= frame["b"][0] <= pitch["length"])
                self.assertTrue(0 <= frame["b"][1] <= pitch["width"])
            if not frame["e"]:
                self.assertEqual(frame["p"], [])
            if frame["h"] is not None:
                self.assertEqual(len(frame["h"]), 9)

    def test_player_stats_and_space_control_are_consistent(self):
        p = self.payload
        duration = p["source"]["duration"]
        ids = {row[0] for frame in p["frames"] for row in frame["p"]}
        for player in p.get("players", []):
            self.assertIn(player["id"], ids)
            self.assertIn(player["team"], ("A", "B", None))
            self.assertIn(player["role"], ("player", "goalkeeper"))
            self.assertGreaterEqual(player["seconds"], 2.0)
            self.assertLessEqual(player["seconds"], duration + 1)
            if player["peak_kmh"] is not None:
                self.assertLessEqual(player["peak_kmh"], 12 * 3.6)
            self.assertLessEqual(player["avg_kmh"], 12 * 3.6)
            self.assertGreaterEqual(player["distance_m"], 0)
            self.assertEqual(len(player["kmh_per_s"]), len(p["space_control"]["series"]))
        space = p.get("space_control")
        if space:
            self.assertEqual(len(space["series"]), int(-(-duration // 1)))
            for row in space["series"]:
                self.assertTrue(row["A"] is None or 0 <= row["A"] <= 1)
                self.assertTrue(0 <= row["support"] <= 1)
            self.assertTrue(0 <= space["supported_fraction"] <= 1)
        model = p.get("model")
        if model:
            self.assertEqual(model["checkpoint_sha256"], p["source"]["model_sha256"])
            for row in model["classes"]:
                self.assertTrue(0 <= row["recall"] <= 1)

    def test_momentum_series_is_bounded_and_contiguous(self):
        series = self.payload["momentum"]
        step = self.payload["bin_seconds"]
        for before, after in zip(series, series[1:]):
            self.assertAlmostEqual(after["t"] - before["t"], step, places=3)
        for item in series:
            if item["value"] is not None:
                self.assertTrue(-1 <= item["value"] <= 1)
                self.assertTrue(0 < item["support"] <= 1)
            else:
                self.assertEqual(item["support"], 0)

    def test_moments_and_notes_are_explained(self):
        duration = self.payload["source"]["duration"]
        for index, moment in enumerate(self.payload["moments"]):
            self.assertEqual(moment["id"], index)
            self.assertIn(moment["type"], MOMENT_TYPES)
            self.assertTrue(moment["title"])
            self.assertTrue(moment["evidence"])
            self.assertIn(moment["confidence"], ("low", "medium"))
            self.assertTrue(0 <= moment["start"] <= moment["end"] <= duration + 1)
            self.assertTrue(0 <= moment["score"] <= 1)
        for index, note in enumerate(self.payload["tactics"]):
            self.assertEqual(note["id"], index)
            self.assertGreaterEqual(note["players_observed"], 1)
            self.assertTrue(note["text"] and note["evidence"])
            self.assertLess(note["start"], note["end"])

    def test_shape_series_has_stats_for_every_team_with_rows(self):
        p = self.payload
        for team, rows in p["shapes"].items():
            self.assertIn(team, p["shape_stats"])
            for row in rows:
                self.assertGreaterEqual(row["n"], 6)
                self.assertTrue(0 <= row["depth"] <= p["pitch"]["length"])
                self.assertTrue(0 <= row["width"] <= p["pitch"]["width"])

    def test_payload_is_strict_json(self):
        text = (DATA / "analysis.json").read_text()
        self.assertNotRegex(text, r"\b(NaN|Infinity)\b")

    def test_page_loads_only_local_assets(self):
        for name in ("index.html", "app.js", "style.css"):
            text = (WEB / name).read_text()
            self.assertNotRegex(text, r"(src|href)=[\"']https?://", name)
            self.assertNotIn("cdn.", text, name)
        html = (WEB / "index.html").read_text()
        for ref in re.findall(r'(?:src|href)="([^"#]+)"', html):
            if not ref.startswith("http"):
                self.assertTrue((WEB / ref).is_file(), ref)


if __name__ == "__main__":
    unittest.main()
