import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from src.data_set.data_set import ExtendedFaceDataset
from src.face_detection.face_detection import FaceDetector
from src.face_recognition.face_recognition import FaceRecognizer, ReportFileManager


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


def run_experiment(percentage: int,  directory: str, save_detection_report: bool) -> None:
    """
    Run a face recognition experiment.

    Args:
        n_components (int): Number of principal components.
        directory (str): Path to the directory containing test images.
    """
    # dataset = ExtendedFaceDataset(n_components, directory)

    dataset_with_face_detection = ExtendedFaceDataset(percentage, directory, FaceDetector(
        'venv/lib/python3.11/site-packages/cv2/data/haarcascade_frontalface_default.xml', save_detection_report), save_detection_report)
    dataset_no_face_detection = ExtendedFaceDataset(percentage, directory)

    run_data_through_model(percentage, dataset_with_face_detection)
    run_data_through_model(percentage, dataset_no_face_detection)


def train_and_test(X_train, y_train, X_test):
    recognizer = FaceRecognizer()
    recognizer.train(X_train, y_train)

    y_pred = recognizer.predict(X_test)
    return y_pred
