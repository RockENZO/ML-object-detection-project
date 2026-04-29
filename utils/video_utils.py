"""
Utility functions for video I/O using OpenCV.
"""
import cv2
from typing import List
import numpy as np


def read_video(video_path: str) -> List[np.ndarray]:
    """
    Read a video file and return a list of frames.
    
    Args:
        video_path: Path to the video file.
    
    Returns:
        List of frames as numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")
    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_video(frames: List[np.ndarray], output_path: str, fps: int = 24) -> None:
    """
    Save a list of frames to a video file.
    
    Args:
        frames: List of frames (numpy arrays) to write.
        output_path: Destination video file path.
        fps: Frames per second for the output video (default: 24).
    """
    if not frames:
        raise ValueError("No frames to save.")
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
