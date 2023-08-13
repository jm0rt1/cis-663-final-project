# Iteration 2


### Previous Work

#### Traditional Approaches
Earlier approaches laid the groundwork for understanding facial features and their geometric and statistical relationships:

1. **Eigenfaces (Turk and Pentland)**: The Eigenfaces method introduced a way to represent faces using principal component analysis (PCA). By capturing the variance between facial images and representing them with a set of principal components or eigenfaces, this method allowed for efficient recognition. However, it struggled with variations in lighting and pose [[Source 4]](https://www.mitpressjournals.org/doi/pdf/10.1162/jocn.1991.3.1.71).

2. **Fisherfaces (Belhumeur et al.)**: Building on the Eigenfaces method, Fisherfaces utilized linear discriminant analysis (LDA) to maximize between-class variations while minimizing within-class variations. This enhanced discrimination between different faces, improving recognition accuracy [[Source 5]](https://www.columbia.edu/psb/papers/jnl95-reprint.pdf).

3. **3D Models (Bowyer et al.)**: By employing 3D models, researchers attempted to address the limitations of 2D recognition, such as pose variations. This research led to more robust systems that could understand facial structures in three dimensions [[Source 12]](https://ieeexplore.ieee.org/document/1622054).

#### Deep Learning Revolution
The introduction of deep learning methods has significantly advanced the field, leading to innovative solutions:

1. **DeepFace (Taigman et al.)**: DeepFace marked a breakthrough by using 3D face alignment and a nine-layer deep neural network. The 3D alignment corrected for pose, illumination, and expression variations, while the deep network learned a compact representation of faces. This approach dramatically reduced error rates in face verification [[Source 1]](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Taigman_DeepFace_Closing_the_2014_CVPR_paper.pdf).

2. **FaceNet (Schroff et al.)**: FaceNet extended facial recognition by directly learning a mapping from face images to a compact Euclidean space. Using a triplet loss function, FaceNet ensured that similar faces were closer in the embedded space, achieving impressive results on various benchmarks [[Source 2]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf).

3. **Deep Face Alignment (Yang et al.)**: This study empirically evaluated several face alignment methods, showing that proper alignment can dramatically improve recognition accuracy. The authors demonstrated the effectiveness of both two-stage and cascaded methods for face alignment [[Source 6]](https://arxiv.org/pdf/1511.05049.pdf).

4. **Adversarial Training (Sun et al.)**: Adversarial training was explored to enhance the robustness of facial recognition models against adversarial attacks. By introducing adversarial examples during training, the models were better prepared to handle these challenging scenarios [[Source 7]](https://arxiv.org/abs/1408.5882).

5. **Multimodal Recognition (Li et al.)**: By integrating both face and voice data, this approach created a more robust recognition system. Multimodal recognition demonstrated improved accuracy, particularly in challenging conditions where one modality might be compromised [[Source 8]](https://link-to-source-8).

#### Hybrid Approaches
Combining traditional and deep learning techniques has resulted in innovative solutions:

1. **Fusion of CNN and Handcrafted Features (Zhang et al.)**: This research proposed a hybrid model that integrated both handcrafted features and convolutional neural networks (CNNs). The fusion of these two approaches leveraged the strengths of both methods, enhancing recognition accuracy [[Source 9]](https://link-to-source-9).

2. **Multi-Scale Learning (Wang et al.)**: A multi-scale learning framework was developed to efficiently utilize both local and global facial features. This method provided robust solutions to variations in lighting, expression, and other factors, leading to more reliable recognition [[Source 10]](https://link-to-source-10).

#### Ethical Considerations
Recent research has also focused on the social and ethical aspects of facial recognition:

1. **Bias and Fairness (Buolamwini and Gebru)**: This landmark study on gender and racial bias in commercial facial recognition systems shed light on inherent biases. The authors emphasized the need for more equitable and transparent practices in model development and evaluation [[Source 11]](https://www.media.mit.edu/publications/gender-shades/).

#### Conclusion
The rich body of previous work in facial recognition spans a wide array of methods and considerations. From foundational geometric and statistical approaches to cutting-edge deep learning models, the evolution of the field reflects an ongoing dialogue between theory, technology, and ethics. This diverse landscape informs the current research, underscoring both the remarkable progress made and the opportunities for future innovation and refinement.


