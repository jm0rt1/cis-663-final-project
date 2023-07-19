# Triplet Loss
Triplet Loss is a loss function used in machine learning and, particularly, in the training of deep neural networks for face recognition tasks. It's designed to improve the accuracy of embeddings by comparing the distance of a positive pair and a negative pair.

Key Points:
- A triplet in this context consists of an anchor example, a positive example that is similar to the anchor, and a negative example that is dissimilar.
- The objective of the Triplet Loss is to ensure that the anchor is closer to the positive example than it is to the negative example by a margin.
- This function is used extensively in training models for face recognition, like FaceNet, where the embeddings of faces of the same person should be closer compared to faces of different people.

- [[../source-notes/Source 2|Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 815-823).]] This is the seminal paper where the concept of Triplet Loss was introduced in the context of face recognition. It introduces the FaceNet model that utilizes Triplet Loss to learn a compact embedding of face images.
