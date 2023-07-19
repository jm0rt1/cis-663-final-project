# Convolutional Neural Networks
Convolutional Neural Networks (CNNs) are a type of artificial neural network designed to process data with a grid-like topology, such as an image, which has the topology of a 2D grid of pixels.

Key Points:
- CNNs are composed of one or more convolutional layers, often followed by pooling layers, and then fully connected layers towards the end of the network.
- They are particularly effective for image recognition tasks because they can automatically and adaptively learn spatial hierarchies of features.
- CNNs have been successfully applied to face recognition tasks, with models like DeepFace using a 9-layer CNN architecture.


## Sources
- [[Source 7|LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.]] This is a foundational paper by Yann LeCun and colleagues that introduces Convolutional Neural Networks and demonstrates their effectiveness in document recognition.

- [[Source 8|Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems (pp. 1097-1105).]] This paper presents the AlexNet model, which significantly popularized the use of CNNs in image recognition tasks with its winning performance in the ImageNet competition.

- [[Source 9|Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., ... & Rabinovich, A. (2015). Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1-9).]] This paper introduces the GoogleNet (or Inception) architecture, which further increased the depth and complexity of CNNs.

- [[../source-notes/Source 1|Taigman, Y., Yang, M., Ranzato, M., & Wolf, L. (2014). DeepFace: Closing the gap to human-level performance in face verification. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1701-1708).]] This paper presents the DeepFace model, which utilizes a 9-layer CNN architecture for face recognition tasks, demonstrating the application of CNNs in the field.

## Code Examples

### Create and train a simple CNN model using Keras 
#CodeExample 
```python
# Import the necessary modules
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the input shape
input_shape = (32, 32, 3) # For RGB images of size 32x32

# Create a sequential model
model = Sequential()

# Add a convolutional layer with 32 filters, 3x3 kernel size, ReLU activation and input shape
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))

# Add a max pooling layer with 2x2 pool size
model.add(MaxPooling2D((2, 2)))

# Add another convolutional layer with 64 filters and 3x3 kernel size
model.add(Conv2D(64, (3, 3), activation='relu'))

# Add another max pooling layer with 2x2 pool size
model.add(MaxPooling2D((2, 2)))

# Add a flatten layer to convert the 3D feature maps to 1D feature vectors
model.add(Flatten())

# Add a dense layer with 64 units and ReLU activation
model.add(Dense(64, activation='relu'))

# Add a dense layer with 10 units and softmax activation for output (assuming 10 classes)
model.add(Dense(10, activation='softmax'))

# Compile the model with categorical crossentropy loss, Adam optimizer and accuracy metric
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model on some training data (X_train, y_train)
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model on some test data (X_test, y_test)
model.evaluate(X_test, y_test)
```