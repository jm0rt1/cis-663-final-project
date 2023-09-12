import unittest
from PIL import Image
import numpy as np
import os

from src.data_set.data_set import ExtendedFaceDataset, FaceDataset


class TestFaceDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = FaceDataset()

    def test_get_data(self):
        images, labels, names = self.dataset.get_data()
        self.assertIsInstance(images, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)
        self.assertIsInstance(names, np.ndarray)
        self.assertGreater(len(images), 0)
        self.assertEqual(len(images), len(labels))
        self.assertEqual(len(names), len(set(labels)))


class TestExtendedFaceDataset(unittest.TestCase):
    def setUp(self):
        # Assuming you have a directory with images labeled as 'true'
        self.true_directory = 'tests/test_files/inputs/tom_cruise'
        self.dataset = ExtendedFaceDataset(
            10, true_directory=self.true_directory)

    def test_get_data(self):
        images, labels, names = self.dataset.get_data()
        label_set = set(labels)
        self.assertIsInstance(images, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)
        self.assertIsInstance(names, list)
        self.assertGreater(len(images), 0)
        self.assertEqual(len(images), len(labels))
        self.assertIn('You', names)
        self.assertEqual(len(names), len(set(labels)))


if __name__ == "__main__":
    unittest.main()
