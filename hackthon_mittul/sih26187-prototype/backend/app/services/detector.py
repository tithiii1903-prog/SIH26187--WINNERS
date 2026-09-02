import torch
from .base_detector import BaseYOLODetector

class PersonDetector(BaseYOLODetector):
    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Initializes the YOLOv8 person detector.
        Automatically checks and uses Apple MPS if available, otherwise CPU.
        """
        super().__init__(model_path)
        
    def detect(self, frame):
        """
        Runs inference on a single OpenCV frame.
        Filters for class 0 (Person).
        Returns a list of dicts: {"box": [x1, y1, x2, y2], "conf": float}
        """
        results = self.model.predict(
            source=frame, 
            device=self.device, 
            classes=[0], 
            verbose=False
        )
        
        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    coords = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    detections.append({
                        "box": [int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])],
                        "conf": conf
                    })
        return detections

    def track(self, frame):
        """
        Runs native tracking inference on a single OpenCV frame using ByteTrack.
        Filters for class 0 (Person).
        Returns a list of dicts: {"box": [x1, y1, x2, y2], "conf": float, "id": int}
        """
        results = self.model.track(
            source=frame, 
            persist=True, 
            device=self.device, 
            classes=[0], 
            tracker="bytetrack.yaml",
            verbose=False
        )
        
        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    coords = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # Track ID may not be present for every box if the tracker hasn't assigned one yet
                    track_id = None
                    if box.id is not None:
                        track_id = int(box.id[0].cpu().numpy())
                        
                    detections.append({
                        "box": [int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])],
                        "conf": conf,
                        "id": track_id
                    })
        return detections

    def reset(self):
        """
        Resets ByteTrack internal tracker state in the YOLO model.
        Clears all Kalman filter states, active tracks, and predictor frame counters
        without reloading or recreating the model weights.
        """
        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            if hasattr(self.model.predictor, "trackers") and self.model.predictor.trackers:
                for tr in self.model.predictor.trackers:
                    if hasattr(tr, "reset"):
                        try:
                            tr.reset()
                        except Exception as e:
                            print(f"[PersonDetector] Error resetting tracker: {e}")
            if hasattr(self.model.predictor, "frame_id"):
                self.model.predictor.frame_id = 0

