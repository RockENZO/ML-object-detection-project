"""Overlapping native-pixel crops for broadcast small-object detection."""

import numpy as np


def tiles(width, height, size=640, overlap=0.2):
    def starts(length):
        last = max(0, length - size)
        step = max(1, int(size * (1 - overlap)))
        return sorted(set([*range(0, last + 1, step), last]))

    return [
        (x, y, min(x + size, width), min(y + size, height))
        for y in starts(height)
        for x in starts(width)
    ]


def suppress(rows, threshold=0.5):
    """Class-aware NMS after translating crop coordinates into the original frame."""
    rows = np.asarray(rows, dtype=float).reshape(-1, 6)
    if not len(rows):
        return rows
    order = np.argsort(-rows[:, 4], kind="stable")
    keep = []
    while len(order):
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        if not len(rest):
            break
        low = np.maximum(rows[i, :2], rows[rest, :2])
        high = np.minimum(rows[i, 2:4], rows[rest, 2:4])
        intersection = np.prod(np.maximum(0, high - low), axis=1)
        area = np.prod(rows[i, 2:4] - rows[i, :2])
        areas = np.prod(rows[rest, 2:4] - rows[rest, :2], axis=1)
        overlap = intersection / np.maximum(area + areas - intersection, 1e-9)
        order = rest[(rows[rest, 5] != rows[i, 5]) | (overlap <= threshold)]
    return rows[keep]


def predict(
    model,
    frame,
    device,
    imgsz=1280,
    confidence=0.1,
    tile_size=640,
    tiled_classes=(0,),
    batch=4,
    tile_confidence=None,
    ball_model=None,
):
    result = model.predict(
        frame, imgsz=imgsz, device=device, conf=confidence, classes=None, verbose=False
    )[0]
    data = result.boxes.data.cpu().numpy()
    cutoff = tile_confidence if tile_confidence is not None else confidence
    rows = data[
        (data[:, 5] != 0) | ((data[:, 4] >= cutoff) & (ball_model is None))
    ].tolist()
    if ball_model is not None:
        full_ball = ball_model.predict(
            frame, imgsz=imgsz, device=device, conf=cutoff, classes=[0], verbose=False
        )[0]
        rows.extend(full_ball.boxes.data.cpu().numpy().tolist())
    tile_model = ball_model if ball_model is not None else model
    if tile_size:
        crops = tiles(frame.shape[1], frame.shape[0], tile_size)
        for start in range(0, len(crops), batch):
            regions = crops[start : start + batch]
            detected = tile_model.predict(
                [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in regions],
                imgsz=imgsz,
                device=device,
                conf=cutoff,
                classes=list(tiled_classes),
                verbose=False,
            )
            for region, small in zip(regions, detected):
                translated = small.boxes.data.cpu().numpy().copy()
                translated[:, [0, 2]] += region[0]
                translated[:, [1, 3]] += region[1]
                rows.extend(translated.tolist())
    import torch

    result.update(
        boxes=torch.as_tensor(
            suppress(rows),
            dtype=result.boxes.data.dtype,
            device=result.boxes.data.device,
        )
    )
    return result
