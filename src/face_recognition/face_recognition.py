import datetime
from pathlib import Path
from typing import Optional
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from src.face_recognition.face_detection import FaceDetector

from src.face_recognition.data_set import ExtendedFaceDataset
from imblearn.over_sampling import SMOTE
from src.shared.settings import GlobalSettings as gs


class FaceRecognizer:
    def __init__(self, n_components: Optional[int] = None) -> None:
        """
        Initialize a face recognizer.

        Args:
            n_components (Optional[int], optional): Number of principal components. Defaults to None.
        """
        self.n_components = n_components
        self.pca: Optional[PCA] = None
        self.clf = SVC(kernel='linear', class_weight='balanced')

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


def run_experiment(percentage: int,  directory: str) -> None:
    """
    Run a face recognition experiment.

    Args:
        n_components (int): Number of principal components.
        directory (str): Path to the directory containing test images.
    """
    # dataset = ExtendedFaceDataset(n_components, directory)
    dataset_with_face_detection = ExtendedFaceDataset(percentage, directory, FaceDetector(
        'venv/lib/python3.11/site-packages/cv2/data/haarcascade_frontalface_default.xml'))
    dataset_no_face_detection = ExtendedFaceDataset(percentage, directory)

    run_data_through_model(percentage, dataset_with_face_detection)
    run_data_through_model(percentage, dataset_no_face_detection)


def run_data_through_model(percentage: int, dataset: ExtendedFaceDataset):
    x, y, target_names = dataset.get_data()
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=42)

    # Use SMOTE only on the training data
    smote = SMOTE(sampling_strategy='auto', k_neighbors=min(2, len(X_train)-1))
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    y_pred = train_and_test(
        X_train, y_train, X_test)

    y_pred_resampled = train_and_test(
        X_train_resampled, y_train_resampled, X_test)

    unique_labels = np.unique(np.concatenate((y_test, y_pred)))
    if len(unique_labels) == 1:
        target_names = ['Not You'] if unique_labels[0] == 0 else ['You']
    else:
        target_names = ['Not You', 'You']

    ReportFileManager().add_classification_report_report_to_file(
        target_names, y_test, y_pred, percentage, False, True if dataset.detector is not None else False)

    ReportFileManager().add_classification_report_report_to_file(
        target_names, y_test, y_pred_resampled, percentage, True, True if dataset.detector is not None else False)


def train_and_test(X_train_resampled, y_train_resampled, X_test_resampled):
    recognizer = FaceRecognizer()
    recognizer.train(X_train_resampled, y_train_resampled)

    y_pred = recognizer.predict(X_test_resampled)
    return y_pred


class ReportFileManager():
    TIME_STAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    REPORT_OUTPUT_DIR = Path(
        gs.OUTPUT_DIR/"reports")
    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE_PATH = REPORT_OUTPUT_DIR/f"report_{TIME_STAMP}.txt"

    report_counter = 1
    header_written = False

    def __init__(self):
        """Initialize a file manager."""
        if not self.header_written:
            with open(ReportFileManager.REPORT_FILE_PATH, 'w') as f:
                f.write(f"Commit ID: {self.get_commit_id()}\n\n")
            ReportFileManager.header_written = True

    def add_classification_report_report_to_file(self, target_names, y_test, y_pred, percentage: float, resampled: bool, face_detection: bool):
        """Output classification report to file."""

        classification_str = classification_report(
            y_test, y_pred, target_names=target_names)
        with open(self.REPORT_FILE_PATH, 'a') as f:
            f.write(
                f"Classification Report #{ReportFileManager.report_counter} -- Percentage of Target in Dataset: {percentage}\nSMOTE Resampled = {resampled}\nFace Detection Used = {face_detection}\n\n")
            f.write(classification_str)
            f.write("\n\n")
        self.report_counter += 1

    @staticmethod
    def get_commit_id():
        """Get the commit ID of the current git repository."""
        import subprocess
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


if __name__ == "__main__":
    print("Running experiment with full LFW dataset...")
    run_experiment(10, "tests/test_files/inputs/tom_cruise")
