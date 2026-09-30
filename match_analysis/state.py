"""Temporal state: measured team colours, bounded prediction, and possession."""

import math
from collections import defaultdict

import numpy as np


class TeamAssigner:
    def __init__(self, margin=0.15, state=None):
        self.margin = margin
        self.centers = (
            np.array(state["centers"], dtype=float)
            if state and state.get("centers")
            else None
        )
        self.samples = []
        self.votes = defaultdict(lambda: np.zeros(2))

    def observe(self, identity, colour):
        if colour is None:
            return None
        colour = np.array(colour, dtype=float)
        if self.centers is None:
            self.samples.append(colour)
            self.samples = self.samples[-300:]
            if len(self.samples) < 120:
                return None
            samples = np.array(self.samples)
            # Warm up across multiple frames and seed from dense colour groups.
            # A stray referee/advertising crop must not seed an entire team.
            distances = np.linalg.norm(samples[:, None] - samples[None, :], axis=2)
            dense = np.flatnonzero(
                np.sum(distances < 0.25, axis=1) >= 0.15 * len(samples)
            )
            if len(dense) < 2:
                return None
            a, b = np.unravel_index(
                np.argmax(distances[np.ix_(dense, dense)]), (len(dense), len(dense))
            )
            centers = samples[dense[[a, b]]]
            for _ in range(20):
                labels = np.argmin(
                    np.linalg.norm(samples[:, None] - centers[None, :], axis=2), axis=1
                )
                if any(np.sum(labels == i) < 0.15 * len(samples) for i in range(2)):
                    return None
                updated = np.array(
                    [np.median(samples[labels == i], axis=0) for i in range(2)]
                )
                if np.allclose(updated, centers):
                    break
                centers = updated
            if np.linalg.norm(centers[0] - centers[1]) < 0.15:
                return None
            self.centers = centers[np.argsort(centers[:, 0])]
        distance = np.linalg.norm(self.centers - colour, axis=1)
        best = int(np.argmin(distance))
        separation = (max(distance) - min(distance)) / max(max(distance), 1e-6)
        if separation < self.margin or min(distance) > 0.45:
            return None
        self.votes[identity][best] += separation
        votes = self.votes[identity]
        return (
            ("A", "B")[int(np.argmax(votes))]
            if max(votes) / sum(votes) >= 0.8 and sum(votes) >= 1
            else None
        )

    def reset_shot(self):
        self.votes.clear()
        self.samples.clear()

    def snapshot(self):
        return {"centers": self.centers.tolist() if self.centers is not None else None}


class Motion:
    def __init__(self, expiry=1.0, max_speed=40.0):
        self.expiry, self.max_speed = expiry, max_speed
        self.history = {}

    def reset(self):
        self.history.clear()

    def observe(self, identity, xy, t):
        if xy is None:
            return False
        xy = np.array(xy, dtype=float)
        prior = self.history.get(identity)
        if (
            prior
            and t > prior[0]
            and np.linalg.norm(xy - prior[1]) / (t - prior[0]) > self.max_speed
        ):
            return False
        velocity = (
            (xy - prior[1]) / (t - prior[0]) if prior and t > prior[0] else np.zeros(2)
        )
        self.history[identity] = (t, xy, velocity)
        return True

    def predict(self, identity, t):
        prior = self.history.get(identity)
        if prior is None or t - prior[0] > self.expiry + 1e-8 or t <= prior[0]:
            return None
        return (prior[1] + prior[2] * (t - prior[0])).tolist()


class Possession:
    def __init__(self, persistence=0.5, radius=2.5, margin=1.0):
        self.persistence, self.radius, self.margin = persistence, radius, margin
        self.reset()

    def reset(self):
        self.candidate = self.stable = None
        self.candidate_since = self.previous_end = None
        self.last_time = None

    def update(self, t, ball, people, eligible, airborne=False):
        """Estimate team control from observations, then compare stable team colours.

        Nearby teammates are evidence for the same team, not a contest. An
        opposing or unassigned nearby player still causes abstention.
        """

        def unknown(reason):
            self.reset()
            self.last_time = t
            self.evidence = {
                "reason": reason,
                "candidate_team": None,
                "support_track_ids": [],
            }
            return "unknown", None

        if not eligible:
            return unknown("excluded_interval")
        if ball is None:
            return unknown("missing_observed_ball")
        if len(ball) != 2 or not all(math.isfinite(v) for v in ball):
            return unknown("invalid_ball_geometry")
        if airborne:
            return unknown("airborne_ball_uncertainty")
        distances = []
        for person in people:
            xy = person.get("xy")
            if (
                xy is not None
                and len(xy) == 2
                and all(math.isfinite(v) for v in xy)
                and person["observation"] == "observed"
                and person.get("role", "player") != "referee"
            ):
                distances.append((math.dist(ball, xy), person))
        distances.sort(key=lambda value: value[0])
        if not distances:
            return unknown("no_observed_players")
        nearest, person = distances[0]
        if nearest > self.radius:
            return unknown("ball_outside_control_radius")
        team = person.get("team")
        if team not in ("A", "B"):
            return unknown("nearest_player_team_unknown")
        contenders = [d for d, p in distances if p.get("team") != team]
        competitor = min(contenders) if contenders else None
        if competitor is not None and competitor - nearest < self.margin:
            return unknown("contested_between_teams")
        if self.last_time is not None and (
            t <= self.last_time or t - self.last_time > 0.25
        ):
            self.reset()
        self.last_time = t
        if team != self.candidate:
            self.candidate, self.candidate_since = team, t
        support = max(0, t - self.candidate_since)
        self.evidence = {
            "reason": "stable_observed_team_control"
            if support + 1e-8 >= self.persistence
            else "awaiting_team_persistence",
            "candidate_team": team,
            "nearest_distance_m": nearest,
            "competing_distance_m": competitor,
            "support_track_ids": [
                p.get("id")
                for d, p in distances
                if p.get("team") == team
                and d <= self.radius
                and p.get("id") is not None
            ],
            "candidate_since": self.candidate_since,
            "observed_support_seconds": support,
        }
        if support + 1e-8 < self.persistence:
            return "unknown", None
        event = None
        if self.stable and self.stable != team:
            event = {
                "type": "possession_change",
                "method": "observed_stable_team_colour_change",
                "from_team": self.stable,
                "to_team": team,
                "timestamp": self.candidate_since,
                "uncertainty": [self.previous_end, self.candidate_since],
                "confirmed_at": t,
                "observed_support_seconds": support,
                "support_track_ids": self.evidence["support_track_ids"],
            }
        self.stable, self.previous_end = team, t
        return team, event


class BallSelector:
    """Select measured candidates using motion evidence; abstain on competing balls."""

    def __init__(self, motion, acquire=0.75, margin=0.15):
        self.motion, self.acquire, self.margin = motion, acquire, margin

    def select(self, candidates, t):
        valid = [c for c in candidates if c.get("xy") is not None]
        if not valid:
            return None, "no_mapped_ball_candidate"
        prior = self.motion.history.get("ball")
        recent = prior is not None and 0 < t - prior[0] <= self.motion.expiry
        if recent:
            dt = t - prior[0]
            limit = self.motion.max_speed * dt + 1.0
            valid = [c for c in valid if math.dist(c["xy"], prior[1]) <= limit]
            if not valid:
                return None, "inconsistent_ball_motion"
            expected = np.asarray(prior[1]) + np.asarray(prior[2]) * dt
            valid.sort(
                key=lambda c: (
                    c["confidence"]
                    - 0.25 * min(math.dist(c["xy"], expected) / max(limit, 1), 2)
                ),
                reverse=True,
            )
        else:
            valid = [c for c in valid if c["confidence"] >= self.acquire]
            if not valid:
                return None, "low_reacquisition_confidence"
            valid.sort(key=lambda c: c["confidence"], reverse=True)
        first = valid[0]
        for other in valid[1:]:
            if (
                math.dist(first["xy"], other["xy"]) > 1
                and abs(first["confidence"] - other["confidence"]) < self.margin
            ):
                return None, "ambiguous_ball_candidates"
        if not self.motion.observe("ball", first["xy"], t):
            return None, "inconsistent_ball_motion"
        return first, "observed_motion_supported" if recent else "observed_reacquired"
