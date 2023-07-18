# processing and manipulation
#CodeExample
```python
# Import the necessary modules
import cv2
import numpy as np

# Load an image from a file
img = cv2.imread('face.jpg')

# Convert the image from BGR to RGB color space
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize the image to 32x32 pixels
img = cv2.resize(img, (32, 32))

# Normalize the pixel values to the range [0, 1]
img = img / 255.0

# Convert the image to a NumPy array
img = np.array(img)

# Print the shape and dtype of the image array
print(img.shape) # (32, 32, 3)
print(img.dtype) # float64
```