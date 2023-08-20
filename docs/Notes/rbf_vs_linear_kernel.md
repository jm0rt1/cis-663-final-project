The choice of kernel in a Support Vector Machine (SVM) has a significant influence on the decision boundary that the SVM learns. When switching from the Radial Basis Function (RBF) kernel to the linear kernel, you are fundamentally changing how the SVM processes and classifies the data. Here are a few reasons why changing to a linear kernel might have helped:

1. **Linear Separability**: If your data is nearly linearly separable or if the true boundary between classes is close to being linear, a linear kernel would perform better. The RBF kernel might overcomplicate the decision boundary, leading to overfitting.

2. **High Dimensionality**: The LFW (Labeled Faces in the Wild) dataset, especially when images are unraveled into long vectors, has a high dimensionality. In high-dimensional spaces, data tends to be more linearly separable, and thus, a linear kernel might perform just as well or even better than more complex kernels.

3. **Overfitting with RBF**: The RBF kernel can be more prone to overfitting, especially if its parameters (like `C` and `gamma`) are not tuned correctly. Overfitting means that while the SVM might perform exceptionally well on training data, it might not generalize well to new, unseen data.

4. **Computational Efficiency**: Linear SVMs are often more computationally efficient than RBF kernel SVMs. Faster training can sometimes lead to more iterations and experimentation, which can help in finding a better model.

5. **Interpretability**: Models using a linear kernel are more interpretable. You can more easily understand the importance of each feature, which can be crucial in certain applications.

6. **Parameter Tuning**: The RBF kernel introduces an additional parameter, `gamma`, that needs tuning. If `gamma` isn't chosen appropriately (either too large or too small), the SVM might not perform well. In contrast, the linear kernel does not have this parameter, which makes tuning a bit simpler.

It's essential to note that the above reasons provide a general overview, and in practice, the performance of linear vs. RBF (or other kernels) depends heavily on the specifics of the data and the problem at hand. Cross-validation and hyperparameter tuning are always recommended to choose the best model and kernel for a particular dataset.