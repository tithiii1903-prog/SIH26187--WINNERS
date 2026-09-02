from .base_detector import BaseYOLODetector

class VehicleDetector(BaseYOLODetector):
    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Initializes the YOLOv8 vehicle detector.
        Automatically checks and uses Apple MPS if available, otherwise CPU.
        """
        super().__init__(model_path)
        # COCO classes for vehicles
        self.vehicle_classes = [2, 3, 5, 7]
        self.class_names = {
            2: "Car",
            3: "Motorcycle",
            5: "Bus",
            7: "Truck"
        }
        
    def detect(self, frame):
        """
        Runs inference on a single OpenCV frame.
        Filters for vehicle classes (2, 3, 5, 7).
        Returns a list of dicts: {"box": [x1, y1, x2, y2], "conf": float, "class_id": int, "class_name": str}
        """
        results = self.model.predict(
            source=frame, 
            device=self.device, 
            classes=self.vehicle_classes, 
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
                    cls_id = int(box.cls[0].cpu().numpy())
                    
                    detections.append({
                        "box": [int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])],
                        "conf": conf,
                        "class_id": cls_id,
                        "class_name": self.class_names.get(cls_id, "Unknown")
                    })
        return detections
