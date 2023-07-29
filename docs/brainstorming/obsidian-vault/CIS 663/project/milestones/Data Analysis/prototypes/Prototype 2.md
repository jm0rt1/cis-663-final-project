# Prototype 2

## Code

Building a robust face recognition system with proper organization and unit tests would be quite a large task that would require designing and implementing several classes and modules, as well as extensive testing. Below is a simplified demonstration of how you might organize your code to adhere to the SOLID principles:

We'll define several classes: `FaceDetector` to handle face detection, `FaceRecognizer` to handle face recognition, and `FaceDataset` to handle the dataset. We'll also define a function `run_experiment` to compare the accuracy of models trained on different numbers of images.

```python
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report


class FaceDetector:
    def __init__(self, cascade_file):
        self.detector = cv2.CascadeClassifier(cascade_file)

    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return [gray[y:y+h, x:x+w] for (x, y, w, h) in faces]


class FaceRecognizer:
    def __init__(self, n_components=150):
        self.pca = PCA(n_components=n_components, whiten=True)
        self.clf = SVC(kernel='rbf', class_weight='balanced')

    def train(self, faces, labels):
        faces_pca = self.pca.fit_transform(faces)
        self.clf.fit(faces_pca, labels)

    def predict(self, face):
        face_pca = self.pca.transform(face)
        return self.clf.predict(face_pca)


class FaceDataset:
    def __init__(self, n_images=None):
        self.dataset = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
        if n_images is not None:
            self.dataset.images, _, self.dataset.target, _ = train_test_split(
                self.dataset.images, self.dataset.target, train_size=n_images, stratify=self.dataset.target, random_state=42)

    def get_data(self):
        n_samples, h, w = self.dataset.images.shape
        images = self.dataset.images.reshape((n_samples, h * w))
        return images, self.dataset.target, self.dataset.target_names


def run_experiment(n_images=None):
    dataset = FaceDataset(n_images)
    X, y, target_names = dataset.get_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    recognizer = FaceRecognizer()
    recognizer.train(X_train, y_train)

    y_pred = recognizer.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=target_names))


if __name__ == "__main__":
    print("Running experiment with 20 images...")
    run_experiment(n_images=20)

    print("Running experiment with full LFW dataset...")
    run_experiment()
```

Please note that this is a relatively simple implementation. For a real-world application, you would likely need to incorporate additional functionality for image preprocessing, model selection and tuning, performance evaluation, and so on.

As for the unit tests, it's generally good practice to write a separate test file for each module. In this case, you might have test files like `test_face_detector.py`, `test_face_recognizer.py`, and `test_face_dataset.py`. Each of these files would contain several functions that test the various methods in the corresponding class. For instance, `test_face_detector.py` might contain a function `test_detect_faces` that checks whether the `detect_faces` method correctly identifies faces in an image.

Given the complexity of the task and the need for data files and other resources, writing comprehensive unit tests would go beyond the scope of this platform. However, I'd highly recommend looking into Python's `unittest` library if you're interested in learning more about unit testing in Python.

### Saving a Trained Model

Absolutely, it's good practice to save a trained model for future use, especially when the training process is time-consuming. You can save your model using Python's built-in `pickle` module.

In your `FaceRecognizer` class, you can add two new methods for saving and loading a trained model:

```python
import pickle

class FaceRecognizer:
    ...
	clf_file_path = ""
	pca_file_path = ""
	
    def save_model(self):
        """
        Save the trained model to a file.
        """
        with open(clf_file_path, 'wb') as f:
            pickle.dump(self.clf, f)
        with open(pca_file_path, 'wb') as f:
            pickle.dump(self.pca, f)
		

    def load_model(self):
        """
        Load a trained model from a file.
        """
        with open(clf_file_path, 'rb') as f:
            self.clf = pickle.load(f)
        with open(pca_file_path, 'rb') as f:
            self.pca = pickle.load(f)
```

Now you can save your trained model after calling `train`, and then load it later to make predictions:

```python
# Train a model
recognizer = FaceRecognizer()
recognizer.train(train_images, train_labels)

# Save the trained model
recognizer.save_model('trained_model.pkl')

# Load the trained model later
recognizer = FaceRecognizer()
recognizer.load_model('trained_model.pkl')

# Make predictions
predictions = recognizer.predict(test_images)
```

Note: This method saves and loads only the classifier (`clf`). If you also want to save and load the PCA transformer (`pca`), you can do so in a similar way, either by modifying the above methods or by creating separate methods for it. 

Also, you should be aware of the security risks associated with using pickle. Only load pickled data that was pickled by trusted sources. If you're sharing your models publicly, consider using a more secure format, like [ONNX](https://onnx.ai/) or [Joblib](https://joblib.readthedocs.io/en/latest/persistence.html).

## Command Line Interface

```python
import argparse
import enum
from face_recognition import FaceRecognizer, load_images
from sklearn.model_selection import train_test_split

class Key(enum.Enum):
    CLI = 'cli'
    EPILOG = 'epilog'
    COMMAND = 'command'
    N_IMAGES = 'n_images'
    MODEL_FILE = 'model_file'

DESCRIPTIONS = {
    Key.CLI: 'A Command Line Interface (CLI) for a face recognition task using the LFW dataset.',
    Key.EPILOG: 'Ensure that you train the model before attempting to make predictions.',
    Key.COMMAND: 'Specify the operation to perform. "train" trains the model on the specified number of images and saves the trained model. "predict" loads the saved model and uses it to predict the identities of the faces in the test set.',
    Key.N_IMAGES: 'Specify the number of images to use for training. If not specified, all images are used.',
    Key.MODEL_FILE: 'Specify the file to save the trained model to or load the trained model from. Default is "model.pkl".',
}

class FaceRecognitionCLI:

    def __init__(self):
        self.parser = argparse.ArgumentParser(description=DESCRIPTIONS[Key.CLI], epilog=DESCRIPTIONS[Key.EPILOG])
        self.parser.add_argument('command', choices=['train', 'predict'], help=DESCRIPTIONS[Key.COMMAND])
        self.parser.add_argument('-n', '--n_images', type=int, help=DESCRIPTIONS[Key.N_IMAGES])
        self.parser.add_argument('-f', '--model_file', default='model.pkl', help=DESCRIPTIONS[Key.MODEL_FILE])

        self.args = self.parser.parse_args()
        self.images, self.labels = load_images()

        if self.args.n_images:
            self.images = self.images[:self.args.n_images]
            self.labels = self.labels[:self.args.n_images]
        
        self.train_images, self.test_images, self.train_labels, self.test_labels = train_test_split(self.images, self.labels, test_size=0.2, random_state=42)
        self.recognizer = FaceRecognizer()

    def train(self):
        self.recognizer.train(self.train_images, self.train_labels)
        self.recognizer.save_model(self.args.model_file)

    def predict(self):
        self.recognizer.load_model(self.args.model_file)
        predictions = self.recognizer.predict(self.test_images)
        print(predictions)

    def run(self):
        if self.args.command == 'train':
            self.train()
        elif self.args.command == 'predict':
            self.predict()

if __name__ == '__main__':
    cli = FaceRecognitionCLI()
    cli.run()
```

You can run this script from the command line like this:

```
python script.py train -n 20 -f my_model.pkl
```

This will train a model using the first 20 images and save it to `my_model.pkl`.

You can then use the trained model to make predictions like this:

```
python script.py predict -f my_model.pkl
```

This will load the model from `my_model.pkl` and use it to predict the identities of the faces in the test set.

Remember to replace `script.py` with the actual name of your script, and `my_model.pkl` with the name you want to give to your model file. Also, you can remove the `-n 20` option if you want to use all images for training.


### Details on the Experiment

The `run_experiment` function is essentially running a face recognition task to compare the accuracy of models trained with different numbers of images. The goal of this experiment is to observe how the quantity of training data can influence the performance of a face recognition system. 

Here's what it does in detail:

1. It loads a dataset of face images. If `n_images` is specified, it randomly selects that many images from the dataset; otherwise, it uses the whole dataset.

2. It splits the dataset into a training set and a test set. The training set is used to train the face recognition model, and the test set is used to evaluate the model's performance.

3. It creates an instance of `FaceRecognizer` and trains it with the training set.

4. It uses the trained model to predict the identities of the faces in the test set.

5. Finally, it prints a classification report, which includes precision, recall, f1-score, and support for each class (each person), as well as overall averages. These are common metrics used to evaluate the performance of classification models.

So when the function is called twice with different arguments (once with `n_images=20` and once with no argument), it effectively runs two experiments: one using only 20 images for training, and one using the entire dataset. By comparing the results of these two experiments, you can gain insights into how the amount of training data affects the face recognition model's performance.






## Unit Tests

Sure, let's write some simple unit tests for each of the classes in the face recognition system. We'll use the `unittest` library from Python's standard library.

The tests here are basic and minimal, mainly demonstrating the pattern of how to structure tests, rather than providing comprehensive coverage. In a real-world scenario, more thorough tests should be considered.

Here is how you might structure your `test_face_detector.py`:

```python
import unittest
import cv2
from face_recognition import FaceDetector

class TestFaceDetector(unittest.TestCase):

    def setUp(self):
        self.detector = FaceDetector('haarcascade_frontalface_default.xml')

    def test_detect_faces(self):
        # Load a test image containing a single face
        image = cv2.imread('test_face.jpg')
        faces = self.detector.detect_faces(image)

        self.assertEqual(len(faces), 1)
        self.assertTrue(isinstance(faces[0], np.ndarray))


if __name__ == "__main__":
    unittest.main()
```

For the `test_face_recognizer.py`:

```python
import unittest
import numpy as np
from face_recognition import FaceRecognizer

class TestFaceRecognizer(unittest.TestCase):

    def setUp(self):
        self.recognizer = FaceRecognizer()

    def test_train(self):
        faces = np.random.rand(10, 1000)
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        self.recognizer.train(faces, labels)

        self.assertIsNotNone(self.recognizer.clf)
        self.assertIsNotNone(self.recognizer.pca)

    def test_predict(self):
        face = np.random.rand(1, 1000)
        label = self.recognizer.predict(face)

        self.assertTrue(label in [0, 1])


if __name__ == "__main__":
    unittest.main()
```

And for `test_face_dataset.py`:

```python
import unittest
from face_recognition import FaceDataset

class TestFaceDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = FaceDataset(20)

    def test_get_data(self):
        images, labels, target_names = self.dataset.get_data()

        self.assertEqual(len(images), 20)
        self.assertEqual(len(labels), 20)
        self.assertTrue(isinstance(target_names, np.ndarray))


if __name__ == "__main__":
    unittest.main()
```

Remember, to run these tests, you would need to have the corresponding files, and in the case of the `FaceDetector` test, an image file `test_face.jpg` that contains a single face.