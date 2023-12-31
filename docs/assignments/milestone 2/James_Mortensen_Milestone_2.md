# Project Proposal

## Group Members:
James Mortensen

## Title: 
Enhancing Facial Recognition Using Advanced Deep Learning Techniques

## Objective: 
The goal of this project is to design and implement a deep learning-based facial recognition system. I will utilize state-of-the-art Convolutional Neural Networks (CNN) architectures, including the innovative techniques presented in the papers by Krizhevsky et al. (2012) and Szegedy et al. (2015), to significantly improve the recognition performance.

## Data Sources:
To train and test our facial recognition system, I will use several publicly available datasets.

Some examples of possible data sets include:

1. **Labeled Faces in the Wild (LFW) Dataset**: This dataset can be used for face verification tasks, where the goal is to determine whether two images depict the same person.
2. **YouTube Faces Database (YTF)**: This dataset, which contains videos instead of static images, may allow this project to test the system's performance under more realistic and challenging conditions.
3. **CelebFaces Attributes (CelebA) Dataset**: This large-scale dataset might be used for both training our system to recognize various celebrities and testing its generalization ability across different attributes and variations.
4. **CASIA-WebFace**: This dataset can provide us with a vast amount of data for training, allowing our system to learn from a large number of subjects.
5. **Face Recognition Grand Challenge (FRGC)**: The project could use use this dataset to benchmark our system and compare its performance with other state-of-the-art systems.
6. **MS-Celeb-1M**: A high-volume dataset that could be used to train our model to recognize a wide array of celebrities.

## Methodology:
The system will first preprocess the images in the datasets, which involves tasks like face detection, alignment, normalization, and augmentation. Then the project will implement and train a CNN model using techniques such as ReLU activation, dropout, overlapping max-pooling, 1x1 convolutions, and auxiliary classifiers. Data parallelism and model parallelism techniques will be employed for efficient training of the large-scale network on distributed systems, as described in the paper by Dean et al. (2012).

The trained model will be evaluated on various tasks like face identification (classifying a face as belonging to one of the known individuals) and face verification (determining whether two face images belong to the same person). 

## Expected Outcome:
The result of this project should be a robust and accurate facial recognition system that demonstrates high performance on various facial recognition tasks and under different conditions. The project will also generate insights into the practical issues and potential solutions in applying advanced deep learning techniques to facial recognition. 

## Future Scope:
In the future, the project can be expanded to explore other applications of facial recognition, such as emotion recognition, age estimation, gender recognition, and so on. Improving the system's privacy and fairness considerations, and deploying the system on different platforms and devices are other possible directions for future work.