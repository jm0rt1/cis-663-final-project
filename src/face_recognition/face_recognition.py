import datetime
from pathlib import Path
from typing import Optional
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report

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


class ReportFileManager():
    TIME_STAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    REPORT_OUTPUT_DIR = Path(
        gs.OUTPUT_DIR/"reports"/"face-recognition"/"classification_reports")
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
