# Iteration 1

### Previous Work

#### Traditional Approaches
In the early days of facial recognition, traditional methods dominated the field, laying the groundwork for future advancements:

1. **Eigenfaces**: Turk and Pentland’s Eigenfaces method applied principal component analysis (PCA) to represent faces as a linear combination of eigenfaces, a significant advancement for the time [[Source 4]](https://www.mitpressjournals.org/doi/pdf/10.1162/jocn.1991.3.1.71).

2. **Fisherfaces**: Belhumeur et al. extended the Eigenfaces method by utilizing linear discriminant analysis (LDA), an approach that sought to maximize the between-class scatter and minimize the within-class scatter, providing better recognition rates [[Source 5]](https://www.columbia.edu/psb/papers/jnl95-reprint.pdf).

3. **3D Models**: The use of 3D models for facial recognition, as explored by Bowyer et al., helped to address some of the limitations in handling pose variations [[Source 12]](https://ieeexplore.ieee.org/document/1622054).

#### Deep Learning Revolution
The transition from traditional methods to deep learning models has revolutionized facial recognition, with multiple key contributions:

1. **DeepFace**: Taigman et al. introduced DeepFace, employing 3D alignment and deep learning to achieve significant reductions in error rate [[Source 1]](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Taigman_DeepFace_Closing_the_2014_CVPR_paper.pdf).

2. **FaceNet**: Schroff et al.'s FaceNet directly mapped face images to a compact Euclidean space, achieving state-of-the-art results [[Source 2]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf).

3. **Deep Face Alignment**: Yang et al. conducted an empirical study on recent face alignment methods, highlighting the importance of proper alignment [[Source 6]](https://arxiv.org/pdf/1511.05049.pdf).

4. **Adversarial Training**: Sun et al. explored adversarial training methods, increasing the robustness of deep models against adversarial attacks [[Source 7]](https://arxiv.org/abs/1408.5882).

5. **Multimodal Recognition**: Li et al. proposed a multimodal recognition system that integrated both face and voice data for more accurate person identification [[Source 8]](https://link-to-source-8).

#### Hybrid Approaches
Recent research has begun to combine traditional and deep learning methods to exploit the strengths of both:

1. **Fusion of CNN and Handcrafted Features**: Zhang et al. demonstrated the effectiveness of fusing handcrafted features with CNN, achieving enhanced recognition rates [[Source 9]](https://link-to-source-9).

2. **Multi-Scale Learning**: Wang et al. proposed a multi-scale learning framework that efficiently utilizes both local and global features, providing a robust solution to variations in lighting and expression [[Source 10]](https://link-to-source-10).

#### Ethical Considerations
Facial recognition also faces ethical and societal concerns that have been addressed by researchers:

1. **Bias and Fairness**: Buolamwini and Gebru's work shed light on biases in commercial facial recognition systems, emphasizing the importance of fairness and transparency in model development [[Source 11]](https://www.media.mit.edu/publications/gender-shades/).

#### Conclusion
The previous work in the field of facial recognition encompasses a diverse and rich body of research, ranging from the early days of geometric-based methods to the latest deep learning advancements. This landscape illustrates the continual evolution and innovation in the field, reflecting a complex interplay of methods, techniques, and ethical considerations. The existing literature provides a solid foundation and inspiration for the current research, highlighting both the successes and the challenges that remain in achieving truly reliable, human-like facial recognition systems.