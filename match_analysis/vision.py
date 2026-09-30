"""Broadcast heuristics and measured image features; no official live/replay guarantee."""

import cv2
import numpy as np


def colour(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    roi = frame[
        y1 + int((y2 - y1) * 0.15) : y1 + int((y2 - y1) * 0.5),
        x1 + int((x2 - x1) * 0.25) : x1 + int((x2 - x1) * 0.75),
    ]
    if roi.size == 0:
        return None
    pixels = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    # Central torso crops reduce background while preserving green team kits.
    if len(pixels) < 10:
        return None
    hsv = np.median(pixels, axis=0)
    # Hue is circular; include saturation and brightness for white/black jerseys.
    angle = hsv[0] * np.pi / 90
    return [
        float(np.cos(angle) * hsv[1] / 255),
        float(np.sin(angle) * hsv[1] / 255),
        float(hsv[2] / 255),
    ]


def airborne_uncertainty(pixel, people):
    """Upper-body image overlap is uncertainty; unrelated foreground heads are not.

    This visual heuristic cannot establish a grounded ball or detect every aerial
    ball. Never describe a negative result as validated ground contact.
    """
    for p in people:
        box = p.get("bbox")
        if not box or p.get("observation") != "observed":
            continue
        x1, y1, x2, y2 = box
        if x1 <= pixel[0] <= x2 and y1 <= pixel[1] < y2 - 0.3 * (y2 - y1):
            return True
    return False


def fingerprint(frame):
    small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (9, 8))
    return int(
        "".join(
            "1" if value else "0" for value in (small[:, 1:] > small[:, :-1]).flatten()
        ),
        2,
    )


class Segmenter:
    def __init__(self, config, history=None):
        self.config = config
        self.previous = None
        self.history = history or []
        self.replay_run = 0

    def update(self, frame, timestamp, player_count):
        hsv = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        cut = (
            self.previous is not None
            and cv2.compareHist(self.previous, hist, cv2.HISTCMP_BHATTACHARYYA)
            > self.config.cut_threshold
        )
        self.previous = hist
        grass = float(
            np.mean((hsv[:, :, 0] > 30) & (hsv[:, :, 0] < 90) & (hsv[:, :, 1] > 40))
        )
        fp = fingerprint(frame)
        matches = [
            item
            for item in self.history
            if timestamp - item["t"] > 10
            and (fp ^ item["hash"]).bit_count() <= self.config.replay_distance
        ]
        self.replay_run = self.replay_run + 1 if matches else 0
        # Sequence repetition is a replay candidate, not a trained replay classifier.
        replay = self.replay_run >= self.config.replay_confirm_samples
        self.history.append({"t": timestamp, "hash": fp})
        self.history = self.history[-600:]
        wide = (
            grass >= self.config.grass_fraction
            and player_count >= self.config.min_players
        )
        label = (
            "replay_candidate" if replay else "wide_candidate" if wide else "unsuitable"
        )
        return {
            "cut": bool(cut),
            "view": label,
            "grass_fraction": grass,
            "eligible_view": bool(wide and not matches),
            "reason": "repeated_visual_content"
            if matches
            else "wide_view_heuristic"
            if wide
            else "insufficient_field_visibility",
        }
