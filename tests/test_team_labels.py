import unittest

from match_analysis.viewer import propagate_team_labels


def frame(rows):
    return {"t": 0.0, "s": 0, "e": 1, "p": rows}


def track(identity, kinds):
    return [frame([[identity, kind, 10.0, 10.0, 1]]) for kind in kinds]


class PropagateTeamLabelsTests(unittest.TestCase):
    def test_unassigned_rows_take_the_tracks_clear_team(self):
        frames = track(1, [2, 2, 0, 0, 0, 0, 0, 0, 2])
        self.assertEqual(propagate_team_labels(frames), 3)
        self.assertEqual({f["p"][0][1] for f in frames}, {0})

    def test_a_split_track_is_left_alone(self):
        frames = track(1, [0, 0, 0, 1, 1, 1, 2, 2])
        self.assertEqual(propagate_team_labels(frames), 0)
        self.assertEqual(frames[-1]["p"][0][1], 2)

    def test_too_little_evidence_changes_nothing(self):
        frames = track(1, [0, 0, 2, 2])
        self.assertEqual(propagate_team_labels(frames), 0)

    def test_referees_goalkeepers_and_other_tracks_are_untouched(self):
        frames = track(1, [0] * 6) + track(2, [2, 2, 2]) + track(3, [4, 4, 4]) + track(4, [3, 3, 3])
        propagate_team_labels(frames)
        kinds = {f["p"][0][0]: f["p"][0][1] for f in frames}
        self.assertEqual(kinds, {1: 0, 2: 2, 3: 4, 4: 3})


if __name__ == "__main__":
    unittest.main()
