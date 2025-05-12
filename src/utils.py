def log_message(message):
    """Logs a message to the console."""
    print(f"[LOG] {message}")

def visualize_detections(image, detections):
    """Visualizes detections on the image."""
    import cv2
    for detection in detections:
        x1, y1, x2, y2, class_id, confidence = detection
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, f"Class: {class_id}, Conf: {confidence:.2f}", 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image

def save_processed_data(data, file_path):
    """Saves processed data to a specified file path."""
    import pickle
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_processed_data(file_path):
    """Loads processed data from a specified file path."""
    import pickle
    with open(file_path, 'rb') as f:
        return pickle.load(f)