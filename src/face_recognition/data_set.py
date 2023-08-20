from typing import List, Tuple
import os
from typing import List, Tuple, Optional
from PIL import Image
from abc import ABC, abstractmethod
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.face_recognition.face_detection import FaceDetector
import cv2


def preprocess_image(img_path_or_array, target_size=(47, 62)):  # Note the switch here
    if isinstance(img_path_or_array, str):
        # If a path is provided, load image
        img = Image.open(img_path_or_array)
    else:
        # If an array is provided, convert back to Image for easy processing
        img = Image.fromarray(img_path_or_array)

    img = img.convert('L')  # Convert to grayscale
    img = img.resize(target_size)  # Resize
    img_array = np.array(img)

    # Normalize pixel values to [0, 1] scale
    normalized_image = img_array / 255.0
    return normalized_image


class BaseFaceDataset(ABC):

    def __init__(self):
        self.images: List[np.array] = []
        self.labels: List[int] = []

    @abstractmethod
    def get_data(self) -> Tuple[np.array, np.array, List[str]]:
        pass


class FaceDataset(BaseFaceDataset):

    def __init__(self, n_images: Optional[int] = None, use_face_detection: bool = True):
        super().__init__()
        self.dataset = fetch_lfw_people(min_faces_per_person=15, resize=0.4)
        self.detector = FaceDetector(
            'venv/lib/python3.11/site-packages/cv2/data/haarcascade_frontalface_default.xml') if use_face_detection else None
        self.use_face_detection = use_face_detection
        self.dataset.target = np.zeros(
            self.dataset.target.shape[0], dtype=np.int32)
        if n_images is not None:
            self.dataset.images, _, self.dataset.target, _ = train_test_split(
                self.dataset.images, self.dataset.target, train_size=min(n_images, len(self.dataset.images)-1), stratify=self.dataset.target, random_state=42)

        # Create a new list to store preprocessed images
        processed_images = []

        # Preprocess the images
        for i in range(self.dataset.images.shape[0]):
            processed_img = preprocess_image(self.dataset.images[i])
            processed_images.append(processed_img)

        # Update the dataset's images attribute
        self.dataset.images = np.array(processed_images)

    def get_data(self) -> Tuple[np.array, np.array, List[str]]:
        images = []
        for image in self.dataset.images:
            if self.use_face_detection:
                faces = self.detector.detect_faces(image)
                if len(faces) == 0:
                    continue
                processed_face = preprocess_image(faces[0])
                images.append(processed_face)
            else:
                processed_image = preprocess_image(image)
                images.append(processed_image)

        if len(images) == 0:
            return np.array([]), np.array([]), []

        n_samples, h, w = len(images), images[0].shape[0], images[0].shape[1]
        images = np.array(images).reshape((n_samples, h * w))

        return images, self.dataset.target, np.array(["Not You"])


class ExtendedFaceDataset(FaceDataset):

    def __init__(self, n_images: Optional[int] = None, true_directory: str = None):
        super().__init__(n_images)
        self.true_directory = true_directory
        if self.true_directory:
            self.inject_true_images()

    def inject_true_images(self):
        target_shape = (62, 47)  # or whatever shape you are aiming for
        for file in os.listdir(self.true_directory):
            if file.lower().endswith('.jpg') or file.lower().endswith('.jpeg'):
                img_data = preprocess_image(
                    os.path.join(self.true_directory, file))

                # Ensure that the image has the correct shape
                if img_data.shape != target_shape:
                    print(
                        f"Skipped image {file} due to inconsistent shape: {img_data.shape}")
                    continue

                img_data = img_data.reshape(-1)  # Flatten the image
                self.images.append(img_data)
                if "true" in file.lower():
                    self.labels.append(1)
                elif "false" in file.lower():
                    self.labels.append(0)
        self.images = np.array(self.images)

    def get_data(self) -> Tuple[np.array, np.array, List[str]]:
        n_samples, h, w = self.dataset.images.shape
        images = self.dataset.images.reshape((n_samples, h * w))
        combined_images = np.vstack((images, self.images))
        false_labels = np.zeros(n_samples, dtype=np.int32)
        combined_labels = np.concatenate((false_labels, self.labels))
        return combined_images, combined_labels, ["Not You", "You"]


class BalancedFaceDataset(ExtendedFaceDataset):

    def get_data(self) -> Tuple[np.array, np.array, List[str]]:
        images, labels, names = super().get_data()

        smote = SMOTE()
        balanced_images, balanced_labels = smote.fit_resample(images, labels)

        return balanced_images, balanced_labels, names


class CustomFaceDataset(BaseFaceDataset):
    def __init__(self, directory: str):
        super().__init__()
        self.names: np.ndarray = np.array(["Other", "You"])

        for file in os.listdir(directory):
            if file.endswith('.jpg'):  # change this if your images are in a different format
                img = Image.open(os.path.join(directory, file))
                img = img.convert('L')  # convert image to grayscale
                # change this if you want a different size
                img = img.resize((62, 47))
                self.images.append(np.array(img))
                if "true" in file.lower():
                    self.labels.append(1)  # label '1' for 'you'
                elif "false" in file.lower():
                    self.labels.append(0)  # label '0' for 'other'

    def get_data(self) -> Tuple[np.array, np.array, List[str]]:
        n_samples = len(self.images)
        h, w = self.images[0].shape
        images = np.array(self.images).reshape((n_samples, h * w))
        return images, np.array(self.labels), self.names


# def preprocess_image(image_path):
#     # Load the image in grayscale
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#     # Normalize pixel values to [0,1] scale
#     normalized_image = image / 255.0

#     return normalized_image


def scale_images(self, images: np.array) -> np.array:
    """
    Scale pixel values of images to range [0, 1] using MinMaxScaler.

    Args:
        images (np.array): Array of images.

    Returns:
        np.array: Scaled images.
    """
    # Instantiate the scaler
    scaler = MinMaxScaler()

    # Reshape the images to 2D array so we can fit the scaler
    images_2d = images.reshape(-1, 1)

    # Fit and transform the data
    images_scaled_2d = scaler.fit_transform(images_2d)

    # Reshape the data back to its original shape
    images_scaled = images_scaled_2d.reshape(images.shape)

    return images_scaled


if __name__ == "__main__":
    # Quick test to ensure everything is working as expected.
    dataset = BalancedFaceDataset(true_directory="path_to_your_images")
    images, labels, names = dataset.get_data()
    print(f"Images Shape: {images.shape}")
    print(f"Labels Shape: {labels.shape}")
    print(f"Names: {names}")
