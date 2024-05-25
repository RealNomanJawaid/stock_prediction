import unittest
import numpy as np
from prediction_based_on_news import createDataset

class TestDatasetCreation(unittest.TestCase):

    def test_create_dataset(self):
        # Define sample data for testing
        sample_features = np.random.rand(10, 5)
        sample_target = np.random.rand(10)
        look_back = 3
        # Create dataset from sample data
        X, y = createDataset(sample_features, sample_target, look_back)
        # Ensure that dataset is created with correct dimensions
        self.assertEqual(X.shape[0], len(sample_target) - look_back)
        self.assertEqual(X.shape[1], look_back)
        self.assertEqual(X.shape[2], sample_features.shape[1])
        self.assertEqual(y.shape[0], len(sample_target) - look_back)

if __name__ == '__main__':
    unittest.main()
