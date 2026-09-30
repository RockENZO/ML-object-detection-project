"""Isolated PnLCalib process avoids its top-level utils/model namespace collisions."""

import argparse
import base64
import contextlib
import json
import sys


def device_string(device):
    # Ultralytics accepts numeric CUDA selectors; torch.Tensor.to needs cuda:N.
    return "cuda:" + str(device) if str(device).isdigit() else device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.device = device_string(args.device)
    sys.path.insert(0, args.root)
    from pathlib import Path

    import cv2
    import numpy as np
    import torch
    import yaml
    from model.cls_hrnet import get_cls_net
    from model.cls_hrnet_l import get_cls_net as get_lines
    from utils.utils_calib import FramebyFrameCalib
    from utils.utils_heatmap import (
        complete_keypoints,
        coords_to_dict,
        get_keypoints_from_heatmap_batch_maxpool,
        get_keypoints_from_heatmap_batch_maxpool_l,
    )

    root = Path(args.root)
    torch.set_num_threads(2)
    models = []
    for filename, config, factory in [
        ("SV_kp", "hrnetv2_w48.yaml", get_cls_net),
        ("SV_lines", "hrnetv2_w48_l.yaml", get_lines),
    ]:
        model = factory(yaml.safe_load((root / "config" / config).read_text()))
        model.load_state_dict(
            torch.load(root / filename, map_location="cpu", weights_only=True)
        )
        model.to(args.device).eval()
        models.append(model)
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            frame = cv2.imdecode(
                np.frombuffer(base64.b64decode(json.loads(line)["image"]), np.uint8),
                cv2.IMREAD_COLOR,
            )
            h, w = frame.shape[:2]
            resized = cv2.cvtColor(cv2.resize(frame, (960, 540)), cv2.COLOR_BGR2RGB)
            tensor = (
                torch.from_numpy(resized.copy())
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .to(args.device)
                / 255
            )
            with torch.inference_mode(), contextlib.redirect_stdout(sys.stderr):
                hm = models[0](tensor)
                hl = models[1](tensor)
                kp = coords_to_dict(
                    get_keypoints_from_heatmap_batch_maxpool(hm[:, :-1]),
                    threshold=0.3434,
                )
                lp = coords_to_dict(
                    get_keypoints_from_heatmap_batch_maxpool_l(hl[:, :-1]),
                    threshold=0.7867,
                )
                points, lines = complete_keypoints(
                    kp[0], lp[0], w=960, h=540, normalize=True
                )
                cam = FramebyFrameCalib(iwidth=w, iheight=h, denormalize=True)
                cam.update(points, lines)
                result = cam.heuristic_voting_ground(refine_lines=True)
            if result is None:
                output = {"valid": False, "reason": "insufficient_pitch_landmarks"}
            else:
                output = {
                    "valid": True,
                    "homography": np.asarray(result["homography"]).tolist(),
                    "error_px": float(result["rep_err"]),
                    "ground_landmarks": len(cam.subsets["ground_plane"]),
                }
            print(json.dumps(output, allow_nan=False), flush=True)
        except Exception as error:
            print(
                json.dumps(
                    {"valid": False, "reason": type(error).__name__ + ": " + str(error)}
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
