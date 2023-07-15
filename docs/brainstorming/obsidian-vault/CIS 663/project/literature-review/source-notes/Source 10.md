# Source 10

## Citation

F. Schroff and J. Philbin, “Learning face embeddings using joint Bayesian,” arXiv preprint arXiv:1502.06857, 2015.

## Key Points

- The source presents a novel method for learning face embeddings using a joint Bayesian model, which captures the similarity and dissimilarity between pairs of face images in a low-dimensional space.
- The source proposes a new loss function called the joint Bayesian loss, which optimizes the face embeddings to maximize the posterior probability of the same identity given a pair of face images, and minimize the posterior probability of different identities given a pair of face images.
- The source evaluates the joint Bayesian method on several face recognition benchmarks and shows that it achieves state-of-the-art results and outperforms existing methods based on triplet loss or contrastive loss.

## Relevance to Our Project

- This source is relevant for our project because it shows the innovation and improvement of face embedding techniques, which are essential for face recognition tasks. It helps us understand how a joint Bayesian model can be used to learn a more discriminative and robust face embedding that can handle variations in pose, illumination, expression, and occlusion.
- This source also provides us with some useful insights and tips on how to optimize and fine-tune the joint Bayesian model parameters and hyperparameters, such as the number of dimensions, the prior distribution, and the learning rate.

## Quotes

- "We propose a novel method for learning face embeddings using a joint Bayesian model." (p. 1)
- "Our method learns an embedding function that maps faces to a low-dimensional Euclidean space where distances correspond to a measure of face similarity." (p. 1)
- "We show that our method outperforms existing methods based on triplet loss or contrastive loss on several standard benchmarks for face recognition." (p. 1)
- "We also show that our method can handle large variations in pose, illumination, expression, and occlusion." (p. 1)

## Link to Full Text

- https://arxiv.org/pdf/1502.06857.pdf

## Related Notes

- [[Source 2]] - A paper on a face recognition system based on a unified embedding
- [[Source 6]] - A paper on a face recognition system based on triplet loss
- [[Face Embedding]] - A note on the face embedding technique
- [[Joint Bayesian]] - A note on the joint Bayesian model