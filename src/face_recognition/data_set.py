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

        # Scale the pixel values to [0, 255] and convert to uint8
        scaled_img_data = np.array([(img * 255).astype(np.uint8)
                                    for img in self.dataset.images])

        # Convert the scaled data to a PIL Image
        self.dataset.images = scaled_img_data

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
