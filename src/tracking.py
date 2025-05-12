# tracking.py

import cv2
import numpy as np

class ObjectTracker:
    def __init__(self, model):
        self.model = model

    def track_objects(self, frame):
        # Perform object detection
        detections = self.model.detect(frame)
        tracked_objects = []

        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection
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