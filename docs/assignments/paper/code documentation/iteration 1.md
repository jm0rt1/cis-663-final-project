# Iteration 1

### `FaceDetector` Class
Located in `face_recognition.py`, the `FaceDetector` class is responsible for detecting faces within an image. It leverages OpenCV's Haar cascades, specifically using the `haarcascade_frontalface_default.xml` file.

#### Relationship to Other Files:
- The detector is used within the main experiment file to identify faces within the provided images, making it a central part of the preprocessing step.

### `FaceRecognizer` Class
Also part of `face_recognition.py`, the `FaceRecognizer` class handles the training and prediction of face labels. It employs principal component analysis (PCA) and a support vector machine (SVM) classifier.

#### Relationship to Other Files:
- This class connects with the experiment's main run file, utilizing the detected faces and performing the recognition task.

### `run_experiment` Function
This function, present in `face_recognition.py`, orchestrates the entire experiment, integrating both the `FaceDetector` and `FaceRecognizer` classes. It is designed to execute the experiment by accepting parameters for the number of components and the image directory.

#### Relationship to Other Files:
- The function serves as the main entry point, combining the functionalities of both classes and executing the experiment.

### Test Cases
In the provided test cases, you have covered different scenarios to ensure that the classes and functions are working as intended. These include:
- `TestFaceDetector`: Tests the `detect_faces` method of the `FaceDetector` class.
- `TestFaceRecognizer`: Tests the `train` and `predict` methods of the `FaceRecognizer` class.
- `TestRunExperiment`: Tests the `run_experiment` function for its execution without errors.

#### Relationship to Other Files:
- These tests are essential for validating the functionality and integration of the various components within the entire system.

### Overall Relation
- `face_recognition.py` is the core file containing the main classes and functions for the face recognition system.
- The test file links to `face_recognition.py`, providing a series of tests to ensure that each piece of functionality is working correctly.

This short documentation should give an overview of the different parts of the code and how they relate to each other. If you need more specific details or elaboration on any part, please let me know!