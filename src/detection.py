from pathlib import Path
import cv2
import torch

class ObjectDetector:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        return model

    def detect_objects(self, image):
        results = self.model(image)
        return results

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        results = self.detect_objects(image)
        return results

    def display_results(self, results, image):
        annotated_image = results.render()[0]
        cv2.imshow('Detection Results', annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    model_path = Path('models/yolov8_weights.pt')  # Update with your model path
    detector = ObjectDetector(model_path)
    
    image_path = Path('data/raw/sample_image.jpg')  # Update with your image path
    results = detector.process_image(image_path)
    detector.display_results(results, image_path)

if __name__ == "__main__":
    main()