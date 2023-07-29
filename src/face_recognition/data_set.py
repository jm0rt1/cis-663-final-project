from typing import List, Tuple
import os
from typing import List, Tuple, Optional
from PIL import Image
from abc import ABC, abstractmethod
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import numpy as np


class BaseFaceDataset(ABC):
    def __init__(self):
        self.images: List[np.array] = []
        self.labels: List[int] = []

    @abstractmethod
    def get_data(self) -> Tuple[np.array, np.array, List[str]]:
        pass


class FaceDataset(BaseFaceDataset):
    def __init__(self, n_images: Optional[int] = None):
        super().__init__()
        self.dataset = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
        if n_images is not None:
            self.dataset.images, _, self.dataset.target, _ = train_test_split(
                self.dataset.images, self.dataset.target, train_size=n_images, stratify=self.dataset.target, random_state=42)

    def get_data(self) -> Tuple[np.array, np.array, List[str]]:
        n_samples, h, w = self.dataset.images.shape
        images = self.dataset.images.reshape((n_samples, h * w))
        return images, self.dataset.target, self.dataset.target_names


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
                self.images.append(img_data)
                self.labels.append(1)  # label '1' for 'true'
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
