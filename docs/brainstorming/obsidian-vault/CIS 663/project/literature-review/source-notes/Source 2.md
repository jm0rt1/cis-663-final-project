# Source 2: FaceNet: A Unified Embedding for Face Recognition and Clustering

## Citation

F. Schroff, D. Kalenichenko and J. Philbin, “FaceNet: A unified embedding for face recognition and clustering,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 815-823, 2015.

## Key Points

- The source presents a novel face recognition system called FaceNet, which uses a deep convolutional neural network (CNN) to learn a unified embedding of face images that is discriminative, compact, and generalizable.
- The source proposes a new loss function called the triplet loss, which optimizes the CNN to minimize the distance between an anchor image and a positive image of the same person, and maximize the distance between the anchor image and a negative image of a different person.
- The source evaluates the FaceNet system on several large-scale face datasets and shows that it achieves state-of-the-art results and surpasses human-level performance.

## Relevance to Our Project

- This source is relevant for our project because it introduces one of the most efficient and effective face recognition systems in the literature. It helps us understand how deep learning can be used to learn a universal representation of face images that can be applied to various tasks such as verification, identification, and clustering.
- This source also provides us with some useful insights into the design and optimization of the CNN architecture and the loss function, which can inspire our further improvement of our own system.

## Quotes

- "We present a system, called FaceNet, that directly learns a mapping from face images to a compact Euclidean space where distances directly correspond to a measure of face similarity." (p. 815)
- "We introduce a novel loss function based on optimizing distances between triplets of images rather than pairs." (p. 816)
- "Our method achieves state-of-the-art accuracy on all of the standard face recognition benchmarks, including LFW (99.63%), YouTube Faces DB (95.12%), and the recently released IARPA Janus Benchmark A (IJB-A) dataset with 91.9% TAR at FAR=0.1%." (p. 821)
- "We also show that we can perform very well on the challenging task of clustering face images from unconstrained photos into identities." (p. 821)

## Link to Full Text

- https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf

## Related Notes

- [[Source 5]] - A paper on a face verification system based on deep learning
- [[Convolutional Neural Networks|CNN]] - A note on the convolutional neural network technique
- [[Triplet Loss]] - A note on the triplet loss function
## Related Notes

- [[Face Embeddings]]
- [[Triplet Loss]]
- [[Large Scale Training]]
- [[Convolutional Neural Networks]]
- [[Facial Recognition Techniques]]