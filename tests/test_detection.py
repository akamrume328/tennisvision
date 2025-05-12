import unittest
from src.detection import load_model, run_inference

class TestDetection(unittest.TestCase):

    def setUp(self):
        # Load the model before each test
        self.model = load_model('path/to/model/config')

    def test_load_model(self):
        # Test if the model loads correctly
        self.assertIsNotNone(self.model)

    def test_run_inference(self):
        # Test the inference function with a sample input
        sample_input = 'path/to/sample/image.jpg'
        result = run_inference(self.model, sample_input)
        self.assertIsInstance(result, dict)  # Assuming the result is a dictionary
        self.assertIn('detections', result)  # Check if 'detections' key exists

if __name__ == '__main__':
    unittest.main()