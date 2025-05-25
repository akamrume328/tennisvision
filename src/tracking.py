# tracking.py

import cv2
import numpy as np

class ObjectTracker:
    def __init__(self, model):
        self.model = model

    def track_objects(self, frame, imgsz=640):
        """
        フレーム内のオブジェクトを検出し追跡します。
        imgsz パラメータを受け取り、モデルの推論に使用します。
        """
        # YOLOモデルの track メソッド (または predict) に imgsz を渡します。
        # 'persist=True' はトラッキングで一般的に使用されます。
        # verbose=False を追加して、コンソール出力を減らすことも検討できます。
        results = self.model.track(frame, persist=True, imgsz=imgsz)
        
        tracked_data = []
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.model.names[cls]
                
                tracked_data.append({
                    'box': (x1, y1, x2, y2),
                    'id': track_id,
                    'confidence': conf,
                    'class_id': cls,
                    'class_name': class_name
                })
        return tracked_data

def draw_tracking_results(frame, tracked_objects):
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj['box']
        track_id = obj['id']
        label = f"ID:{track_id} {obj['class_name']} {obj['confidence']:.2f}"

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame