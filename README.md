# Subspace Indexing on Stiefel and Grassmann Manifolds

**Authors:** Wenqing Hu, Tiefeng Jiang, Birendra Kathariya, Vikram Abrol, Jiali Zhang and Zhu Li

**Published at:** IEEE BigData 2023 Conference

---

## Overview

This work addresses the challenge of nonlinear representation learning in high-dimensional data. While classic linear methods (PCA, LDA) fail to capture local variations and nonlinear methods (kernel algorithms, DNNs) are computationally expensive, we propose the **Subspace Indexing Model with Interpolation (SIM-I)** — a lightweight, locality-aware approach that achieves comparable performance to deep neural networks while maintaining computational efficiency and theoretical interpretability through piecewise linear, globally nonlinear embeddings.

## Abstract

Classic linear methods like Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) struggle to capture the nonlinearity and local variations in complex, high-dimensional data. While powerful nonlinear methods like Kernel Algorithms, Manifold Learning, and Deep Neural Networks (DNNs) exist, they are often computationally very expensive.

**Research Question:** How can we design a lightweight, locality-aware model that achieves nonlinear representation learning comparable to deep neural networks, while maintaining computational efficiency and theoretical interpretability?

This paper proposes the **Subspace Indexing Model with Interpolation (SIM-I)**, which builds a piecewise linear, locality-aware, yet globally nonlinear embedding.

## Architecture

![LIE Architecture](LIE%20build%20from%20SIM-I%20interpreted%20as%20a%20shallow%20network.png)

The SIM-I model can be interpreted as a shallow neural network with locality-aware linear projections in the first hidden layer, followed by selective activation and weighted aggregation in the second hidden layer for KNN classification.

---

## Code Structure

### (a) MATLAB Code

- **`Stiefel_Optimization.m`**
  Object class for optimization calculus and differential geometry on Stiefel manifolds, including tangent projection, exponential map, geodesics, gradient descent, retraction, lifting, logarithmic map, etc.

- **`Grassmann_Optimization.m`**
  Object class for optimization calculus and differential geometry on Grassmann manifolds, including tangent projection, exponential map, geodesics, gradient descent, retraction, lifting, logarithmic map, etc.

- **`buildVisualWordList.m`**
  Partitions a given sample data set according to a tree of given height into leaf nodes.

- **`SIFT_PCA.m`**
  Performs SIFT (Scale Invariant Feature Transform) PCA analysis.

- **`SIFT_PCA_Recovery.m`**
  SIFT PCA recovery using the Stiefel_Optimization method, compared with benchmark nearest neighbor method.

- **`LPP_CenterMass.m`**
  Classification analysis based on Laplacian eigenface and graph Laplacian method, as well as center of mass on Grassmann manifold. Applied to datasets: nwpu-aerial-images, MNIST, cifar10.

### (b) Python Code

- **`Stiefel_Optimization.py`**
  Python implementation with same functionality as the MATLAB version.

- **`Grassmann_Optimization.py`**
  Python implementation with same functionality as the MATLAB version.

- **`buildVisualWordList.py`**
  Python implementation with same functionality as the MATLAB version.

- **`LPP_CenterMass.py`**
  Classification analysis based on Laplacian eigenface and graph Laplacian method, as well as center of mass on Grassmann manifold. Applied to datasets: MNIST, CIFAR-10. Incorporates GMM sampling of pseudo-data inputs and labeling by pre-trained model.

- **`LPP_Auxiliary.py`**
  Auxiliary functions for Laplacian eigenface and graph Laplacian method, including: k-nearest neighbor, graph Laplacian, supervised affinity, LPP generalized eigenvalue problem.

- **`cifar10vgg.py`**
  Builds a pre-trained VGG model for CIFAR-10, can also train a new model. Pre-trained model parameters available at [cifar-vgg](https://github.com/geifmany/cifar-vgg).

- **`umap_data_aug.py`**
  Generates new pseudo data points based on current dataset using UMAP and 2-simplices.

- **`MNISTLeNetv2.py`**
  Builds a pre-trained LeNet-v2 model for MNIST.

- **`vox1VggFace.py`**
  Builds a pre-trained VGG model for face datasets.
