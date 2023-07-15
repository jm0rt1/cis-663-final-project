# Source 9

## Citation

C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov et al., “Going deeper with convolutions,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 1-9, 2015.

## Key Points

- The source presents a novel deep convolutional neural network (CNN) architecture called Inception, which is inspired by the idea of network-in-network and uses multiple parallel filters of different sizes and dimensions to capture more complex and diverse features from the input images.
- The source proposes several techniques to improve the efficiency and scalability of the Inception network, such as using 1x1 convolutions to reduce the dimensionality of the feature maps, using batch normalization to speed up the training process, and using auxiliary classifiers to regularize the network and boost the gradient signal.
- The source evaluates the Inception network on several large-scale image recognition tasks, such as ImageNet classification, ImageNet detection, and COCO detection, and shows that it achieves state-of-the-art results and surpasses human-level performance.

## Relevance to Our Project

- This source is relevant for our project because it shows the innovation and improvement of deep learning on computer vision and image recognition. It helps us understand how a more sophisticated and modular CNN can be designed and trained to achieve higher accuracy and robustness on various tasks.
- This source also provides us with some useful insights and tips on how to optimize and fine-tune the Inception network parameters and hyperparameters, which can help us improve our own system.

## Quotes

- "We propose a deep convolutional neural network architecture codenamed “Inception”, which was responsible for setting the new state of the art for classification and detection in the ImageNet Large-Scale Visual Recognition Challenge 2014 (ILSVRC14)." (p. 1)
- "The main idea of the Inception architecture is based on finding out how an optimal local sparse structure in a convolutional vision network can be approximated and covered by readily available dense components." (p. 2)
- "We have shown that our Inception architecture improves over previous state-of-the-art networks in terms of quality while also reducing computational cost." (p. 8)

## Link to Full Text

- https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf

## Related Notes

- [[Source 8]] - A paper on a deep CNN for ImageNet classification
- [[CNN]] - A note on the convolutional neural network technique
- [[Inception]] - A note on the Inception network architecture