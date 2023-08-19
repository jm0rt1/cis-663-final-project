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
        # set labels to zeros of the same length
        self.dataset.target = np.zeros(
            self.dataset.target.shape[0], dtype=np.int32)
        if n_images is not None:
            self.dataset.images, _, self.dataset.target, _ = train_test_split(
                self.dataset.images, self.dataset.target, train_size=n_images, stratify=self.dataset.target, random_state=42)

    def get_data(self) -> Tuple[np.array, np.array, List[str]]:
        images = []
        for image in self.dataset.images:
            if self.use_face_detection:
                faces = self.detector.detect_faces(image)
                # if no faces are detected, continue to the next image
                if len(faces) == 0:
                    continue
                # assuming that detect_faces returns a list of detected faces,
                # you may want to just use the first detected face
                images.append(faces[0])
            else:
                images.append(image)

        # If no images were added, return empty arrays
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
        for file in os.listdir(self.true_directory):
            if file.endswith('.jpg'):  # change this if your images are in a different format
                img = Image.open(os.path.join(self.true_directory, file))
                img = img.convert('L')  # convert image to grayscale
                # resize image to match LFW images size
                img = img.resize((37, 50))
                img_data = np.array(img).reshape(-1)  # flatten the image
                # Instantiate the scaler
                scaler = MinMaxScaler()
                # Rescale the images
                # save original dimension
                original_dimension = img_data.shape[0]
                img_data = scaler.fit_transform(img_data.reshape(-1, 1))
                # Reshape the image back to its original dimensions
                img_data = img_data.reshape(original_dimension)
                self.images.append(img_data)
                if "true" in file.lower():
                    self.labels.append(1)  # label '1' for 'you'
                elif "false" in file.lower():
                    self.labels.append(0)  # label '0' for 'other'
        self.images = np.array(self.images)

    def get_data(self) -> Tuple[np.array, np.array, List[str]]:
        # Overriding this function to label other faces as 'false'
        n_samples, h, w = self.dataset.images.shape
        images = self.dataset.images.reshape((n_samples, h * w))
        # Combine LFW images and your 'true' images
        combined_images = np.vstack((images, self.images))
        # Label all LFW images as 'false'
        # use np.int32 or np.int64
        false_labels = np.zeros(n_samples, dtype=np.int32)
        # Combine labels
        combined_labels = np.concatenate((false_labels, self.labels))
        return combined_images, combined_labels, np.array(["Not You"] + ['You'])


class BalancedFaceDataset(ExtendedFaceDataset):
    """
    This dataset attempts to balance the classes using SMOTE oversampling.
    """

    pass


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


def preprocess_image(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Normalize pixel values to [0,1] scale
    normalized_image = image / 255.0

    return normalized_image


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
