# Source 11

## Citation

Dean, J., Corrado, G., Monga, R., Chen, K., Devin, M., Le, Q. V., ... & Ng, A. Y. (2012). Large scale distributed deep networks. In Advances in neural information processing systems (pp. 1223-1231).

## Key Points

- The source presents a novel framework for training large-scale deep neural networks (DNNs) on distributed clusters of machines, using a combination of model and data parallelism techniques.
- The source introduces several innovations and techniques to improve the scalability and efficiency of the distributed DNN training, such as using asynchronous stochastic gradient descent (ASGD) to update the model parameters, using a parameter server architecture to store and communicate the parameters, and using a DistBelief system to manage the computation and communication tasks.
- The source evaluates the distributed DNN framework on several large-scale machine learning tasks, such as speech recognition, image recognition, and natural language processing, and shows that it achieves state-of-the-art results and reduces the training time significantly.

## Relevance to Our Project

- This source is relevant for our project because it shows the feasibility and benefit of training large-scale DNNs on distributed systems, which can overcome some of the limitations of single-machine training such as memory constraints, computational bottlenecks, and data availability.
- This source also provides us with some useful insights and tips on how to design and implement a distributed DNN framework, which can help us scale up our own system.

## Quotes

- "We have developed a software framework called DistBelief that can utilize computing clusters with thousands of machines to train large models." (p. 1223)
- "We have successfully used this framework to train models with more than one billion parameters on problems with more than 100 million data samples." (p. 1223)
- "We have found that using larger models and larger datasets can greatly improve generalization performance across a variety of domains." (p. 1224)
- "We have also found that using asynchronous stochastic gradient descent can significantly speed up training time without sacrificing accuracy." (p. 1224)

## Link to Full Text

- https://papers.nips.cc/paper/2012/file/6aca97005c68f1206823815f66102863-Paper.pdf

## Related Notes

- [[Source 8]] - A paper on a deep CNN for ImageNet classification
- [[Source 9]] - A paper on a novel CNN architecture called Inception
- [[DNN]] - A note on the deep neural network technique
- [[ASGD]] - A note on the asynchronous stochastic gradient descent technique