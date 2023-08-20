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
        detected_faces = []
        for angle in [0, 90, 180, 270]:  # Rotations to check
            rotated_img = self.rotate_image(image, angle)
            gray = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)

            if angle == 0:  # We'll just display the original grayscale image, not all rotations
                matplotlib_face(gray)

            faces = self.detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(400, 400))

            if isinstance(faces, np.ndarray) and len(faces) > 0:
                # Assuming 'gray' is the grayscale image you're working on
                matplotlib_faces(gray, faces)

            for (x, y, w, h) in faces:
                aspect_ratio = w / h
                if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                    detected_faces.append(gray[y:y+h, x:x+w])

        return detected_faces

    @staticmethod
    def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
        """
        Rotate an image by a given angle.

        Args:
            image (np.ndarray): Image to rotate.
            angle (int): Rotation angle in degrees. Positive values mean counter-clockwise rotation.

        Returns:
            np.ndarray: Rotated image.
        """
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))


def matplotlib_face(face: np.ndarray):
    import matplotlib.pyplot as plt

    plt.imshow(face, cmap='gray')
    plt.show()


def matplotlib_faces(original_img: np.ndarray, face_coords: np.ndarray):
    import matplotlib.pyplot as plt

    # Extract faces from original image using bounding box coordinates
    extracted_faces = [original_img[y:y+h, x:x+w]
                       for (x, y, w, h) in face_coords]

    n_faces = len(extracted_faces)

    if n_faces == 1:
        fig, axs = plt.subplots()
        axs.imshow(extracted_faces[0], cmap='gray')
    else:
        fig, axs = plt.subplots(n_faces)
        for i, face in enumerate(extracted_faces):
            axs[i].imshow(face, cmap='gray')
    plt.show()
