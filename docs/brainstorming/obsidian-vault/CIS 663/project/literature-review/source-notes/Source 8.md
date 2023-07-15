# Source 8

## Citation

Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems (pp. 1097-1105).

## Key Points

- The source presents a novel deep convolutional neural network (CNN) architecture that achieves state-of-the-art results on the ImageNet Large Scale Visual Recognition Challenge (ILSVRC), which is a large-scale image classification task with 1000 classes and over 1 million images.
- The source introduces several innovations and techniques to improve the performance and efficiency of the CNN, such as using rectified linear units (ReLU) as activation functions, using dropout as a regularization method, using overlapping max-pooling layers, and using data augmentation to increase the size and diversity of the training set.
- The source also demonstrates the generalization ability of the CNN by applying it to other image recognition tasks, such as object detection and face recognition, and shows that it outperforms existing methods.

## Relevance to Our Project

- This source is relevant for our project because it shows the breakthrough and impact of deep learning on computer vision and image recognition. It helps us understand how a large and complex CNN can be designed and trained to achieve high accuracy and robustness on a challenging and realistic task.
- This source also provides us with some useful insights and tips on how to optimize and fine-tune the CNN parameters and hyperparameters, which can help us improve our own system.

## Quotes

- "We trained one of the largest convolutional neural networks to date on the subsets of ImageNet used in the ILSVRC-2010 and ILSVRC-2012 competitions. We obtained error rates of 37.5% and 17.0% on the respective test sets, while the best published result in the literature is 26.2% from a much shallower network." (p. 1097)
- "Our network contains a number of new and unusual features which improve its performance and reduce its training time, which are detailed in Section 3. Some of these features include: (1) ReLU Nonlinearity; (2) Training on Multiple GPUs; (3) Local Response Normalization; (4) Overlapping Pooling; (5) Dropout." (p. 1098)
- "The power of our network is illustrated by its performance on two tasks outside of ILSVRC: PASCAL VOC object detection and a human face recognition dataset." (p. 1104)

## Link to Full Text

- https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

## Related Notes

- [[Source 7]] - A survey on gradient-based learning applied to document recognition
- [[Convolutional Neural Networks|CNN]] - A note on the convolutional neural network technique
- [[ImageNet]] - A note on the ImageNet dataset and challenge