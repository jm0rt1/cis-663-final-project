import unittest
import numpy as np
import cv2
# Import your classes here
from src.face_recognition.face_recognition import FaceDetector, FaceRecognizer, run_experiment


class TestFaceDetector(unittest.TestCase):
    def setUp(self):
        # replace with your cascade_file
        self.detector = FaceDetector(
            'venv/lib/python3.11/site-packages/cv2/data/haarcascade_frontalface_default.xml')
        # replace with your test image file
        self.image = cv2.imread(
            'tests/test_files/inputs/tom_cruise/true1.jpeg')

    def test_detect_faces(self):
        faces = self.detector.detect_faces(self.image)
        self.assertTrue(isinstance(faces, list))  # check if output is a list
        # check if all items in list are np.ndarray
        self.assertTrue(all(isinstance(i, np.ndarray) for i in faces))


class TestFaceRecognizer(unittest.TestCase):
    def setUp(self):
        # set n_components to match data
        self.recognizer = FaceRecognizer(n_components=10)
        # now we have 50 samples and 100 features
        self.faces = np.random.rand(50, 100)
        self.labels = np.array([0, 1]*25)  # adjust labels to match faces

    def test_train(self):
        self.recognizer.train(self.faces, self.labels)
        # Check if classifier was trained (clf is not None)
        self.assertIsNotNone(self.recognizer.clf)

    def test_predict(self):
        self.recognizer.train(self.faces, self.labels)
        prediction = self.recognizer.predict(self.faces[0].reshape(1, -1))
        # Check if prediction returns np.ndarray
        self.assertTrue(isinstance(prediction, np.ndarray))
        # Check if prediction is within possible labels
        self.assertTrue(prediction[0] in [0, 1])


class TestRunExperiment(unittest.TestCase):
    def test_run_experiment(self):
        # Here we simply check if run_experiment runs without throwing any error.
        # You may wish to write more sophisticated tests checking the printed output or other side effects.
        try:
            run_experiment(10, "tests/test_files/inputs/tom_cruise")
            experiment_runs = True
        except:
            experiment_runs = False

        self.assertTrue(experiment_runs)


if __name__ == '__main__':
    unittest.main()
