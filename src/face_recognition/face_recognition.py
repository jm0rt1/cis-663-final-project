from typing import Optional
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report

from src.face_recognition.data_set import ExtendedFaceDataset, FaceDataset


class FaceDetector:
    def __init__(self, cascade_file: str) -> None:
        """
        Initialize a face detector with a given cascade file.

        Args:
            cascade_file (str): Path to the cascade file.
        """
        self.detector = cv2.CascadeClassifier(cascade_file)

    def detect_faces(self, image: np.ndarray) -> list:
        """
        Detect faces in an image.

        Args:
            image (np.ndarray): Image in which to detect faces.

        Returns:
            list: List of detected face regions.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return [gray[y:y+h, x:x+w] for (x, y, w, h) in faces]


class FaceRecognizer:
    def __init__(self, n_components: Optional[int] = None) -> None:
        """
        Initialize a face recognizer.

        Args:
            n_components (Optional[int], optional): Number of principal components. Defaults to None.
        """
        self.n_components = n_components
        self.pca: Optional[PCA] = None
        self.clf = SVC(kernel='rbf', class_weight='balanced')

    def train(self, faces: np.ndarray, labels: np.ndarray) -> None:
        """
        Train the face recognizer.

        Args:
            faces (np.ndarray): Training face images.
            labels (np.ndarray): Labels corresponding to the faces.
        """
        n_components = min(
            faces.shape[0], faces.shape[1]) if self.n_components is None else self.n_components
        self.pca = PCA(n_components=n_components, whiten=True)
        faces_pca = self.pca.fit_transform(faces)
        self.clf.fit(faces_pca, labels)

    def predict(self, face: np.ndarray) -> np.ndarray:
        """
        Predict the label for a given face.

        Args:
            face (np.ndarray): Face for which to predict the label.

        Returns:
            np.ndarray: Predicted label.
        """
        face_pca = self.pca.transform(face)
        return self.clf.predict(face_pca)


def run_experiment(n_components: int, directory: str) -> None:
    """
    Run a face recognition experiment.

    Args:
        n_components (int): Number of principal components.
        directory (str): Path to the directory containing test images.
    """
    dataset = ExtendedFaceDataset(n_components, directory)

    X, y, target_names = dataset.get_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    recognizer = FaceRecognizer()
    recognizer.train(X_train, y_train)

    y_pred = recognizer.predict(X_test)

    unique_labels = np.unique(np.concatenate((y_test, y_pred)))
    if len(unique_labels) == 1:
        target_names = ['Not You'] if unique_labels[0] == 0 else ['You']
    else:
        target_names = ['Not You', 'You']

    print(classification_report(y_test, y_pred, target_names=target_names))


if __name__ == "__main__":
    print("Running experiment with full LFW dataset...")
    run_experiment(10, "tests/test_files/inputs/tom_cruise")
