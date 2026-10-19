import json
import tempfile
import unittest
from pathlib import Path

from match_analysis.config import Config
from match_analysis.control import recompute, track_teams


def person(identity, team, role="player", observation="observed"):
    return {
        "id": identity,
        "role": role,
        "team": team,
        "xy": [0, 0],
        "bbox": [0, 0, 10, 20],
        "observation": observation,
        "confidence": 0.9,
    }


def write_run(root, teams_by_frame):
    source = Path(root) / "source"
    (source / "shots").mkdir(parents=True)
    (source / "checkpoint.json").write_text(
        json.dumps({"complete": True, "team_state": {"centers": [[0.1, 0.2, 0.9], [0.0, 0.0, 0.9]]}})
    )
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "video_end": len(teams_by_frame) / 10,
                "signature": {"config": Config().as_dict(), "metadata": {}},
            }
        )
    )
    records = [
        {
            "frame_index": i,
            "timestamp": i / 10,
            "shot": 0,
            "eligible": True,
            "people": [person("0:1", team), person("0:2", None, role="referee")],
            "ball": {"xy": [0, 0], "observation": "observed"},
            "possession": "unknown",
            "event": None,
        }
        for i, team in enumerate(teams_by_frame)
    ]
    (source / "shots" / "000000.jsonl").write_text("".join(json.dumps(f) + "\n" for f in records))
    return source


class TrackTeamTests(unittest.TestCase):
    def test_clear_tracks_get_a_team_and_weak_ones_do_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            clear = write_run(tmp, [None, None] + ["A"] * 8)
            self.assertEqual(track_teams(clear), {(0, "0:1"): "A"})
        with tempfile.TemporaryDirectory() as tmp:
            split = write_run(tmp, ["A", "A", "A", "B", "B", "B", None])
            self.assertEqual(track_teams(split), {})
        with tempfile.TemporaryDirectory() as tmp:
            short = write_run(tmp, [None, "A", "A", None])
            self.assertEqual(track_teams(short), {})

    def test_predicted_rows_are_not_votes(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = write_run(tmp, ["A"] * 4)
            shot = source / "shots" / "000000.jsonl"
            rows = [json.loads(line) for line in shot.read_text().splitlines()]
            for row in rows:
                row["people"][0]["observation"] = "predicted"
            rows.extend(rows[:2])
            shot.write_text("".join(json.dumps(r) + "\n" for r in rows))
            self.assertEqual(track_teams(source), {})

    def test_recompute_fills_unassigned_rows_only_when_asked(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = write_run(tmp, [None] * 6 + ["A"] * 10)
            recompute(source, Path(tmp) / "plain")
            recompute(source, Path(tmp) / "filled", propagate_teams=True)
            read = lambda name: [  # noqa: E731
                json.loads(line)
                for line in (Path(tmp) / name / "shots" / "000000.jsonl").read_text().splitlines()
            ]
            plain, filled = read("plain"), read("filled")
            self.assertEqual([f["people"][0]["team"] for f in plain[:6]], [None] * 6)
            self.assertEqual([f["people"][0]["team"] for f in filled[:6]], ["A"] * 6)
            self.assertTrue(all(f["people"][1]["team"] is None for f in filled))
            kept = json.loads((Path(tmp) / "filled" / "checkpoint.json").read_text())
            self.assertEqual(kept["team_state"], {"centers": [[0.1, 0.2, 0.9], [0.0, 0.0, 0.9]]})
            manifest = json.loads((Path(tmp) / "filled" / "manifest.json").read_text())
            self.assertEqual(manifest["team_labels_propagated"], {"enabled": True, "player_rows_relabelled": 6})


if __name__ == "__main__":
    unittest.main()
