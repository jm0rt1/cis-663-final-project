# Source 2: FaceNet: A Unified Embedding for Face Recognition and Clustering

## Citation

F. Schroff, D. Kalenichenko and J. Philbin, "FaceNet: A unified embedding for face recognition and clustering," _2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_, Boston, MA, USA, 2015, pp. 815-823, doi: 10.1109/CVPR.2015.7298682.

## Key Points

1. **Introduction of FaceNet**: The authors present FaceNet, a system that directly learns a mapping from face images to a compact Euclidean space where distances directly correspond to a measure of face similarity.

2. **Triplet Loss Function**: The authors use a novel triplet loss function to train the network. The triplet consists of two matching face thumbnails and one non-matching face thumbnail.

3. **Large Scale Training**: The model is trained on a large dataset of 200 million images of 8 million identities.

4. **Superior Performance**: The authors demonstrate that the learned embeddings can be used for face recognition, verification, and clustering tasks, outperforming many state-of-the-art methods.

## Critical Data or Evidence

1. **Performance Metrics**: The FaceNet model achieves a new record accuracy of 99.63% on the LFW dataset, and 95.12% on the YouTube Faces DB.
   
2. **Robustness**: Through various evaluations, the authors illustrate that FaceNet embeddings are robust to pose, lighting, and expression variations.

## Relevance to Our Project

1. **Novel Approach**: The introduction of FaceNet provides us with an innovative perspective on tackling face recognition tasks. The idea of learning an embedding from face images to a compact Euclidean space where distances directly correspond to face similarity offers us a new avenue for model design.

2. **Performance Benchmarking**: The excellent performance of FaceNet sets a high benchmark for face recognition systems and can guide our expectations and performance targets for our own project.

## Quotes

1. _"We present a system, called FaceNet, that directly learns a mapping from face images to a compact Euclidean space where distances directly correspond to a measure of face similarity."_ (p. 815)
2. _"In contrast to most current methods, which learn and extract features from the images, our system trains a network to directly optimize the embedding itself."_ (p. 817)

## Link to Full Text

[Full Text](https://ieeexplore.ieee.org/document/7298682)

## Related Notes

- [[Face Embeddings]]
- [[Triplet Loss]]
- [[Large Scale Training]]
- [[Convolutional Neural Networks]]
- [[Facial Recognition Techniques]]
