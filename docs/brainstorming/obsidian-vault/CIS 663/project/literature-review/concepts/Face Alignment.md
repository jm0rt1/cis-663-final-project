# Face Alignment

Face alignment is the process of detecting and aligning facial landmarks on a given face image. Facial landmarks are points of interest on a face, such as the corners of the eyes, the tip of the nose, the mouth contour, etc. Face alignment is an important step for many face-related applications, such as face recognition, face verification, face editing, and face animation.

Face alignment can be divided into two categories: holistic methods and local methods. Holistic methods try to align the whole face region using global transformations, such as affine or projective transformations. Local methods try to align each facial landmark individually using local transformations, such as deformable models or regression models.

Holistic methods are usually faster and simpler than local methods, but they may not be able to handle large variations in pose, expression, and occlusion. Local methods are usually more accurate and robust than holistic methods, but they may require more computational resources and training data.

Some examples of holistic methods are:

- [[Eigenfaces]] [[Source 4]](https://www.mitpressjournals.org/doi/pdf/10.1162/jocn.1991.3.1.71): This method uses principal component analysis (PCA) to extract a low-dimensional representation of face images, and then uses it to align new faces by finding the optimal projection coefficients.
- [[DeepFace]] [[Source 1]](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Taigman_DeepFace_Closing_the_2014_CVPR_paper.pdf): This method uses a deep convolutional neural network (CNN) to learn a high-level representation of face images, and then uses it to align new faces by finding the optimal 3D transformation parameters.

Some examples of local methods are:

- Active Shape Models (ASM) [[Source 3]](https://ieeexplore.ieee.org/document/381842): This method uses a statistical model of facial shape variations, and then uses it to align new faces by iteratively fitting the model to the image using gradient descent.
- [[FaceNet]] [[Source 2]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf) [[Source 2]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf): This method uses a deep CNN to learn a low-dimensional embedding of face images, and then uses it to align new faces by finding the nearest neighbors in the embedding space.
- [[OverFeat]] [[Source 5]](https://arxiv.org/pdf/1312.6229.pdf): This method uses a deep CNN to perform multi-scale detection and localization of facial landmarks, and then uses them to align new faces by applying local affine transformations.