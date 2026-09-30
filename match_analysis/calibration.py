import base64
import json
import selectors
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from .hashing import digest


class PitchMapper:
    def __init__(self, root, device, config):
        self.config = config
        root = Path(root).resolve()
        assets = json.loads((Path(__file__).parent / "pnl_assets.json").read_text())
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        if (
            subprocess.call(["git", "-C", str(root), "diff", "--quiet", "HEAD"])
            or revision != assets["commit"]
        ):
            raise ValueError("Unpinned PnLCalib source")
        for name, item in assets["weights"].items():
            if digest(root / name) != item["sha256"]:
                raise ValueError("Calibration weights differ from verified manifest")
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(Path(__file__).parent / "pnl_worker.py"),
                "--root",
                str(root),
                "--device",
                device,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            if not self._receive().get("ready"):
                raise RuntimeError("Calibration worker failed to start")
        except Exception:
            self.close()
            raise
        self.reset()

    def _receive(self):
        if not self.selector.select(120):
            self.close()
            raise TimeoutError("Calibration worker timeout")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Calibration worker exited; see diagnostic stderr")
        return json.loads(line)

    def reset(self):
        self.matrix = None
        self.gray = None
        self.last_fit = -1e9
        self.last_valid = -1e9

    def update(self, frame, timestamp):
        _, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        method = "unknown"
        error = None
        if timestamp - self.last_fit >= 1 / self.config.calibration_hz - 1e-8:
            self.last_fit = timestamp
            _, image = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            self.process.stdin.write(
                json.dumps({"image": base64.b64encode(image).decode()}) + "\n"
            )
            self.process.stdin.flush()
            result = self._receive()
            error = result.get("error_px")
            if (
                result["valid"]
                and error is not None
                and error <= self.config.max_reprojection_px * w / self.config.imgsz
                and result.get("ground_landmarks", 0) >= 4
            ):
                matrix = np.array(result["homography"])
                scale = np.diag(
                    [self.config.pitch_length / 105, self.config.pitch_width / 68, 1.0]
                )
                # PnLCalib world coordinates are centred at midfield.
                corner = np.array([[1.0, 0.0, 52.5], [0.0, 1.0, 34.0], [0.0, 0.0, 1.0]])
                self.matrix = scale @ corner @ matrix
                self.last_valid = timestamp
                method = "landmarks"
            else:
                self.matrix = None
                method = "unknown"
        elif self.matrix is not None and self.gray is not None:
            pts = cv2.goodFeaturesToTrack(
                self.gray, maxCorners=120, qualityLevel=0.02, minDistance=12
            )
            if pts is not None:
                new, ok, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, pts, None)
                backward, ok_back, _ = (
                    cv2.calcOpticalFlowPyrLK(gray, self.gray, new, None)
                    if new is not None
                    else (None, None, None)
                )
                if backward is not None:
                    good = (
                        (ok.flatten() == 1)
                        & (ok_back.flatten() == 1)
                        & (np.linalg.norm(backward - pts, axis=2).flatten() < 1.5)
                    )
                    if np.sum(good) >= 8:
                        transform, inliers = cv2.findHomography(
                            pts[good], new[good], cv2.RANSAC, 3
                        )
                        if (
                            transform is not None
                            and abs(np.linalg.det(transform)) > 1e-10
                            and np.mean(inliers) >= 0.6
                        ):
                            self.matrix = self.matrix @ np.linalg.inv(transform)
                            method = "optical_flow"
            if method == "unknown":
                self.matrix = None
        if timestamp - self.last_valid > self.config.prediction_seconds:
            self.matrix = None
            method = "unknown"
        self.gray = gray
        valid = (
            self.matrix is not None
            and np.all(np.isfinite(self.matrix))
            and abs(np.linalg.det(self.matrix)) > 1e-12
        )
        if not valid:
            self.matrix = None
        return {
            "valid": bool(valid),
            "method": method,
            "error_px": error,
            "homography": self.matrix.tolist() if valid else None,
        }

    def project(self, point):
        if self.matrix is None:
            return None
        value = self.matrix @ np.array([*point, 1.0])
        if abs(value[2]) < 1e-8:
            return None
        xy = value[:2] / value[2]
        if (
            not np.all(np.isfinite(xy))
            or not 0 <= xy[0] <= self.config.pitch_length
            or not 0 <= xy[1] <= self.config.pitch_width
        ):
            return None
        return xy.tolist()

    def close(self):
        if hasattr(self, "process"):
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if hasattr(self, "selector"):
            self.selector.close()
