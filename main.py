#!/usr/bin/env python3
"""
Main entry point for ML Object Detection Project.
Detects football players in videos using YOLOv5 for training and YOLOv8 for inference.
"""
import argparse
import sys
from pathlib import Path

from utils.video_utils import read_video, save_video
from tracker.tracker import Tracker


def process_video(input_path: str, output_path: str, model_path: str) -> None:
    """
    Process a video file: read frames, run object detection tracking, and save output.
    
    Args:
        input_path: Path to input video file.
        output_path: Path where output video will be saved.
        model_path: Path to YOLO model weights.
    """
    print(f"Reading video from {input_path}...")
    video_frames = read_video(input_path)
    if not video_frames:
        print("Error: No frames read from video.", file=sys.stderr)
        sys.exit(1)
    print(f"Read {len(video_frames)} frames.")

    print(f"Loading tracker with model {model_path}...")
    tracker = Tracker(model_path)

    print("Running object detection and tracking...")
    detections = tracker.detect_frames(video_frames)
    print(f"Got {len(detections)} detection batches.")

    # For now, just save the original frames (detection visualization can be added later)
    # TODO: Annotate frames with bounding boxes from detections
    print(f"Saving video to {output_path}...")
    save_video(video_frames, output_path)
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Football player detection in videos.")
    parser.add_argument("--input", type=str, default="input_videos/08fd33_4.mp4",
                        help="Path to input video (default: input_videos/08fd33_4.mp4)")
    parser.add_argument("--output", type=str, default="output_videos/output_video.avi",
                        help="Path to output video (default: output_videos/output_video.avi)")
    parser.add_argument("--model", type=str, default="models/yolov5su.pt",
                        help="Path to YOLO model (default: models/yolov5su.pt)")
    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    process_video(args.input, args.output, args.model)


if __name__ == "__main__":
    main()
