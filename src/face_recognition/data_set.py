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


def show_image(img_path_or_array):
    if isinstance(img_path_or_array, str):
        # If a path is provided, load image
        img = Image.open(img_path_or_array)
    else:
        # scale up to 0,255 if normalized to 0,1
        if np.max(img_path_or_array) <= 1:
            img_path_or_array = img_path_or_array * 255
        # If an array is provided, convert back to Image for easy processing
        img = Image.fromarray(img_path_or_array)
    img.show()


class BaseFaceDataset(ABC):

    def __init__(self):
        self.images: List[np.array] = []
        self.labels: List[int] = []

    @abstractmethod
    def get_data(self) -> Tuple[np.array, np.array, List[str]]:
        pass


class FaceDataset(BaseFaceDataset):

    def __init__(self, n_images: Optional[int] = None, face_detector: Optional[FaceDetector] = None):
        super().__init__()
        self.dataset = fetch_lfw_people(min_faces_per_person=15, resize=0.4)

        # Scale the pixel values to [0, 255] and convert to uint8
        scaled_img_data = np.array([(img * 255).astype(np.uint8)
                                    for img in self.dataset.images])

        # Convert the scaled data to a PIL Image
        self.dataset.images = scaled_img_data

        self.detector = face_detector
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
            if self.detector:
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

    def __init__(self, desired_percentage: float, true_directory: Optional[str] = None, face_detector: Optional[FaceDetector] = None):
        count = sum("true" in file for file in os.listdir(true_directory))
        n_images = calculate_n_components_to_add_from_lfw(
            count, desired_percentage)
        self.desired_percentage = desired_percentage
        super().__init__(n_images, face_detector)
        self.true_directory = true_directory
        if self.true_directory:
            self.inject_true_images()

    def inject_true_images(self):
        target_shape = (62, 47)  # or whatever shape you are aiming for
        dir_list = os.listdir(self.true_directory)
        for file in dir_list:
            if file.lower().endswith('.jpg') or file.lower().endswith('.jpeg'):

                if self.detector:
                    img = np.array(Image.open(
                        os.path.join(self.true_directory, file)))

                    faces = self.detector.detect_faces(img)
                    # for face in faces:

                    #     face = Image.fromarray(face)
                    #     face.show()

                    if len(faces) == 0:
                        continue
                    img_data = preprocess_image(faces[0])
                else:
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
        n_samples = min(calculate_n_components_to_add_from_lfw(
            self.images.shape[0], self.desired_percentage), n_samples)

        images = self.dataset.images[0:n_samples].reshape((n_samples, h * w))
        combined_images = np.vstack((images, self.images))
        false_labels = np.zeros(n_samples, dtype=np.int32)
        combined_labels = np.concatenate((false_labels, self.labels))

        return combined_images, combined_labels, ["Not You", "You"]


def calculate_n_components_to_add_from_lfw(directory_true_count: int, desired_percentage: int) -> int:
    """
    Calculate the number of components to add to the dataset to achieve a desired percentage of true images.

    Args:
        directory_true_count (int): Number of true images in the directory.
        desired_percentage (int): Desired percentage of true images.

    Returns:
        int: Rounded number of components to add.
    """
    n_components_to_add = directory_true_count / (desired_percentage / 100)
    n_components_to_add -= directory_true_count
    return round(n_components_to_add)
