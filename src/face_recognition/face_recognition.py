import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report

from src.data_set import CustomFaceDataset, FaceDataset


class FaceDetector:
    def __init__(self, cascade_file):
        self.detector = cv2.CascadeClassifier(cascade_file)

    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
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


def run_experiment(directory=None):
    if directory is None:
        dataset = FaceDataset()
    else:
        dataset = CustomFaceDataset(directory)
    X, y, target_names = dataset.get_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    recognizer = FaceRecognizer()
    recognizer.train(X_train, y_train)

    y_pred = recognizer.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=target_names))


if __name__ == "__main__":

    print("Running experiment with full LFW dataset...")
    run_experiment()
