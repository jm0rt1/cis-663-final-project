# Eigenfaces

- Eigenfaces is a technique for face recognition that applies PCA to lower the dimension of face images and express them as weighted sums of a set of basic images known as eigenfaces .
- Eigenfaces are obtained by calculating the eigenvectors of the covariance matrix of the feature vectors of a big set of training face images . The eigenvectors represent the directions of greatest variation in the face image space and capture the most significant differences in facial appearance.
- Eigenfaces can be used to encode and compare new face images by projecting them onto the subspace defined by the eigenfaces and computing the projection coefficients . The coefficients indicate how much each eigenface contributes to the approximation of the original image.
- Eigenfaces have some benefits such as efficiency, compactness, and simplicity . They can represent face images using a few numbers and perform recognition tasks using simple distance measures. They can also deal with variations in illumination, expression, and pose to some degree.
- Eigenfaces also have some drawbacks such as sensitivity to alignment, occlusion, and background . They require precise preprocessing of face images to align the eyes, nose, and mouth. They also presume that faces are frontal and unoccluded. They may not be able to differentiate faces that have similar global appearance but different local features.

 [[Source 4|Turk, M., & Pentland, A. (1991). Eigenfaces for recognition. Journal of cognitive neuroscience, 3(1), 71-86.]] [PDF](https://www.mitpressjournals.org/doi/pdf/10.1162/jocn.1991.3.1.71)

 [[Source 3|Chellappa, R., Wilson, C. L., & Sirohey, S. (1995). Human and machine recognition of faces: a survey. Proceedings of the IEEE, 83(5), 705-741.]] [Face Recognition] [Survey]  [PDF](https://ieeexplore.ieee.org/document/381842)

 [Eigenfaces - Scholarpedia](http://www.scholarpedia.org/article/Eigenfaces)