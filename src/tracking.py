# tracking.py

import cv2
import numpy as np

class ObjectTracker:
    def __init__(self, model):
        self.model = model

    def track_objects(self, frame):
        # Perform object detection
        results = self.model(frame, verbose=False)  # verbose=False を追加
        tracked_objects = []

        # Process results if detections are present
        if results and results[0].boxes:  # Check if results[0] and results[0].boxes exist
            for box in results[0].boxes:  # Iterate through detected boxes
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get coordinates
                confidence = float(box.conf[0])  # Get confidence score
                class_id = int(box.cls[0])  # Get class ID
                
                tracked_objects.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'class_id': class_id
                })

        return tracked_objects

def draw_tracking_results(frame, tracked_objects):
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj['bbox']
        confidence = obj['confidence']
        class_id = obj['class_id']

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {class_id}, Conf: {confidence:.2f}', 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame