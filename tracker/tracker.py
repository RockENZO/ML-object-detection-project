"""
Object tracking module using YOLO and supervision.
"""
from typing import List, Any
import ultralytics
import supervision as sv


class Tracker:
    """
    A tracker that uses a YOLO model for detection and supervision's ByteTracker for tracking.
    """

    def __init__(self, model_path: str) -> None:
        """
        Initialize the Tracker with a YOLO model.
        
        Args:
            model_path: Path to the YOLO model weights (e.g., 'models/yolov5su.pt').
        """
        self.model: ultralytics.YOLO = ultralytics.YOLO(model_path)
        self.tracker = sv.ByteTracker()

    def detect_frames(self, frames: List[Any], batch_size: int = 20, conf: float = 0.1) -> List[Any]:
        """
        Run object detection on batches of frames.
        
        Args:
            frames: List of video frames (numpy arrays).
            batch_size: Number of frames per batch.
            conf: Confidence threshold for detections.
            
        Returns:
            List of detection results (one per batch).
        """
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            # Use track method if available, otherwise predict
            if hasattr(self.model, 'track'):
                det_batch = self.model.track(batch, conf=conf)
            else:
                det_batch = self.model.predict(batch, conf=conf)
            detections.extend(det_batch)
        return detections

    def get_object_tracks(self, frames: List[Any]) -> List[sv.Detections]:
        """
        Get object tracks for a list of frames using the YOLO model and ByteTracker.
        
        Args:
            frames: List of video frames.
            
        Returns:
            List of supervision Detections objects, one per frame.
        """
        detections = self.model.predict(frames)
        tracks = []
        for idx, detection in enumerate(detections):
            cls_names = detection.names
            # Convert to supervision Detections
            det_supervision = sv.Detections.from_ultralytics(detection)
            # Update tracker
            tracked = self.tracker.update_with_detections(det_supervision)
            tracks.append(tracked)
            # Optional: print for debugging
            if idx % 50 == 0:
                print(f"Frame {idx}: {len(tracked)} tracks")
        return tracks
