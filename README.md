# Tennis Vision Project

## Overview
The Tennis Vision Project is an application designed to analyze tennis matches using computer vision techniques. By leveraging YOLOv8 and OpenCV, this project aims to detect and track players, balls, and court lines in video footage, providing insights into player performance and match dynamics.

## Project Structure
```
tennis-vision-project
├── data
│   ├── processed          # Processed datasets ready for model training and evaluation
│   └── raw               # Raw data, such as original video files or images
├── models                 # Directory for storing trained model files
├── notebooks              # Jupyter notebooks for data processing and model training
│   ├── 1_data_preprocessing.ipynb
│   └── 2_model_training.ipynb
├── scripts                # Python scripts for data preprocessing and model training
│   ├── preprocess_data.py
│   └── train.py
├── src                    # Source code for the application
│   ├── __init__.py
│   ├── config.py         # Configuration settings for the project
│   ├── detection.py      # Functions and classes for object detection
│   ├── main.py           # Main entry point of the application
│   ├── tracking.py       # Functions and classes for object tracking
│   └── utils.py          # Utility functions used across the project
├── tests                  # Unit tests for the application
│   ├── __init__.py
│   ├── test_detection.py
│   └── test_tracking.py
├── README.md              # Documentation for the project
└── requirements.txt       # List of dependencies required for the project
```

## Setup Instructions
1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd tennis-vision-project
   ```

2. **Create a virtual environment:**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```
   pip install -r requirements.txt
   ```

## Usage Guidelines
- **Data Preprocessing:** Use the Jupyter notebook `notebooks/1_data_preprocessing.ipynb` to load and preprocess the raw data.
- **Model Training:** Train the model using `notebooks/2_model_training.ipynb`, adjusting hyperparameters as necessary.
- **Running the Application:** Execute the main application by running `src/main.py`, which will orchestrate the workflow of data preprocessing, model training, and evaluation.

## Contribution
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License - see the LICENSE file for details.