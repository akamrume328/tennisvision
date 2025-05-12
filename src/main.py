# main.py

import os
from src.config import Config
from src.detection import load_model, run_inference
from src.tracking import track_objects
from scripts.preprocess_data import preprocess_data
from scripts.train import train_model

def main():
    # Load configuration settings
    config = Config()

    # Step 1: Data Preprocessing
    print("Starting data preprocessing...")
    preprocess_data(config.raw_data_path, config.processed_data_path)

    # Step 2: Model Training
    print("Starting model training...")
    model = load_model(config.model_weights_path)
    train_model(model, config.processed_data_path, config.training_params)

    # Step 3: Object Tracking
    print("Starting object tracking...")
    video_path = os.path.join(config.raw_data_path, "sample_video.mp4")  # Example video path
    track_objects(video_path, model)

if __name__ == "__main__":
    main()