import os
import cv2
import numpy as np
import pandas as pd

def load_raw_data(data_dir):
    raw_data = []
    for filename in os.listdir(data_dir):
        if filename.endswith(('.mp4', '.jpg', '.png')):
            file_path = os.path.join(data_dir, filename)
            raw_data.append(file_path)
    return raw_data

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    # Resize image to a fixed size
    image = cv2.resize(image, (640, 480))
    # Normalize image
    image = image / 255.0
    return image

def save_processed_data(processed_data, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, image in enumerate(processed_data):
        save_path = os.path.join(save_dir, f'processed_image_{i}.npy')
        np.save(save_path, image)

def main(raw_data_dir, processed_data_dir):
    raw_data = load_raw_data(raw_data_dir)
    processed_data = [preprocess_image(image_path) for image_path in raw_data]
    save_processed_data(processed_data, processed_data_dir)

if __name__ == "__main__":
    raw_data_directory = '../data/raw'  # Adjust path as necessary
    processed_data_directory = '../data/processed'  # Adjust path as necessary
    main(raw_data_directory, processed_data_directory)