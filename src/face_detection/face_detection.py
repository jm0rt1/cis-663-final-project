import cv2
import numpy as np
import pathlib
import shutil
from src.face_detection.report import generate_html_report
from src.shared.settings import GlobalSettings as gs

FACE_DETECTOR_REPORT_DIR = gs.OUTPUT_DIR / \
    "reports"/"face-detection"
DETECTED_FACES_OUTPUT_DIR = FACE_DETECTOR_REPORT_DIR/"detected-faces"
SOURCE_IMAGES = FACE_DETECTOR_REPORT_DIR/"source-images"


class FaceDetector:
    def __init__(self, cascade_file: str, generate_report: bool = False, report_output_dir: pathlib.Path = DETECTED_FACES_OUTPUT_DIR) -> None:
        """
        Initialize a face detector with a given cascade file.

        Args:
            cascade_file (str): Path to the cascade file.
            generate_report (bool): Whether to generate an HTML report for detected faces.
            report_output_dir (str): Directory to save the face images and HTML report.
        """
        self.detector = cv2.CascadeClassifier(cascade_file)
        self.generate_report = generate_report
        self.report_output_dir = report_output_dir
        if self.generate_report:
            self.report_data = []
            self.faces_idx = 0
            self.sources_idx = 0

            self.report_output_dir.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(self.report_output_dir, ignore_errors=True)
            shutil.rmtree(SOURCE_IMAGES, ignore_errors=True)

    def detect_faces(self, image: np.ndarray, min_aspect_ratio=0.75, max_aspect_ratio=1.3, show=False) -> list:
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

            if angle == 0 and show:  # We'll just display the original grayscale image, not all rotations
                matplotlib_face(gray)

            faces = self.detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(400, 400))

            if isinstance(faces, np.ndarray) and len(faces) > 0 and show:
                # Assuming 'gray' is the grayscale image you're working on
                matplotlib_faces(gray, faces)

            detected_faces = self.__filter_by_aspect_ratio(
                min_aspect_ratio, max_aspect_ratio, detected_faces, gray, faces)

        self.save_images(detected_faces, gray)

        return detected_faces

    def __filter_by_aspect_ratio(self, min_aspect_ratio, max_aspect_ratio, detected_faces, gray, faces):
        for (x, y, w, h) in faces:
            aspect_ratio = w / h
            if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                detected_faces.append(gray[y:y+h, x:x+w])
        return detected_faces

    def save_images(self, detected_faces, source_image):
        if self.generate_report:
            saved_faces = self.save_detected_faces(detected_faces)
            source_image_path = self.save_source_image(source_image)

            # generate_html_report(
            #     saved_faces, FACE_DETECTOR_REPORT_DIR/"out.html")

            self.report_data.append({
                "source_image": source_image_path,
                "detected_faces": saved_faces
            })

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

    def save_source_image(self, image: np.ndarray) -> str:
        """
        Save the source image to the specified directory.

        Returns:
            str: Path to the saved source image.
        """
        SOURCE_IMAGES.mkdir(parents=True, exist_ok=True)

        source_image_path = SOURCE_IMAGES / \
            f"source_image{self.sources_idx}.png"
        self.sources_idx += 1
        cv2.imwrite(source_image_path.as_posix(), image)
        return source_image_path

    def save_detected_faces(self, detected_faces: list) -> list[pathlib.Path]:
        """
        Save detected faces to the specified directory.

        Args:
            detected_faces (list): List of detected face images.
            output_dir (str): Directory to save the face images.

        Returns:
            list: List of paths to the saved face images.
        """

        DETECTED_FACES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        self.saved_face_paths: list[pathlib.Path] = []
        for idx, face in enumerate(detected_faces):
            face_filename = DETECTED_FACES_OUTPUT_DIR / \
                f"face_{self.faces_idx}.png"
            cv2.imwrite(face_filename.as_posix(), face)  # type:ignore
            self.saved_face_paths.append(face_filename)
            self.faces_idx += 1

        return self.saved_face_paths

    def finalize_report(self) -> None:
        """
        Generate an HTML report showcasing all the source images and their detected faces.
        """
        html_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Detected Faces Report</title>
                <style>
                    img {
                        margin: 10px;
                        border: 1px solid #ddd;
                        border-radius: 4px;
                        padding: 5px;
                        width: 150px;
                    }
                </style>
            </head>
            <body>
                {report_content}
            </body>
            </html>
            """

        report_content = ""
        for item in self.report_data:
            report_content += f'<h2>Source Image</h2><img src="{item["source_image"]}" alt="Source Image" width="500" /><br>'
            report_content += '<h2>Detected Faces</h2>'
            for idx, path in enumerate(item["detected_faces"]):
                report_content += f'<img src="{path}" alt="Detected Face {idx}" />'
            report_content += '<hr>'

        with open(FACE_DETECTOR_REPORT_DIR/"faces_report.html", 'w') as file:
            file.write(html_template.replace(
                "{report_content}", report_content))


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
