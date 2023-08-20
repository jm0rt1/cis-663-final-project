import cv2
import numpy as np


class FaceDetector:
    def __init__(self, cascade_file: str) -> None:
        """
        Initialize a face detector with a given cascade file.

        Args:
            cascade_file (str): Path to the cascade file.
        """
        self.detector = cv2.CascadeClassifier(cascade_file)

    def detect_faces(self, image: np.ndarray, min_aspect_ratio=0.75, max_aspect_ratio=1.3) -> list:
        """
        Detect faces in an image.

        Args:
            image (np.ndarray): Image in which to detect faces.

        Returns:
            list: List of detected face regions.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        matplotlib_face(gray)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(400, 400))
        if isinstance(faces, np.ndarray) and len(faces) > 0:
            matplotlib_faces(faces)
        filtered_faces = []
        for (x, y, w, h) in faces:
            aspect_ratio = w / h
            if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                filtered_faces.append(gray[y:y+h, x:x+w])

        return filtered_faces


def matplotlib_face(face: np.ndarray):
    import matplotlib.pyplot as plt

    plt.imshow(face, cmap='gray')
    plt.show()


def matplotlib_faces(faces: np.ndarray[np.ndarray]):

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(len(faces))
    for i, face in enumerate(faces):
        axs[i].imshow(face, cmap='gray')
    plt.show()
