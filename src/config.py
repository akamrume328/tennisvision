# Configuration settings for the Tennis Vision Project

import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Model directory
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Training settings
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001

# YOLOv8 model settings
YOLOV8_MODEL_PATH = os.path.join(MODEL_DIR, 'yolov8_weights.pt')
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# Logging settings
LOGGING_LEVEL = 'INFO'  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

# Other settings
USE_GPU = True  # Set to False to use CPU
