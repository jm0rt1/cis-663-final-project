# Preprocessing Techniques

1. **Pixel Normalization**: This is where pixel values are scaled to the range 0-1 by dividing by the maximum pixel value, which is usually 255. This method is often used when working with neural networks as they prefer small input values.
    
2. **Standardization (Z-Score Normalization)**: This is where pixel values are scaled so that they have a zero mean and unit variance. This method is often used in traditional machine learning methods, but it can also be beneficial for deep learning models, especially in combination with batch normalization layers.
    
3. **Centering**: This involves subtracting the mean pixel value from each pixel in the image. This operation is often used when working with pre-trained models that were trained on centered data, such as VGG or ResNet models trained on ImageNet data.
    
4. **Contrast Normalization**: This method involves scaling the pixels such that the resultant distribution of pixel values has a standard deviation of 1. It can be beneficial for images with poor lighting conditions.