<b>Subspace indexing on Stiefel and Grassmann manifolds</b>

a project by Wenqing Hu, Tiefeng Jiang and Li Zhu

<b>(a) folder "matlab_code"</b>

(1) Stiefel_Optimization.m 

the object class for optimization calculus and differential geometry on Stiefel manifolds, such as tangent projection, exponential map, geodesics, gradient descent, retraction, lifting, logarithmic map, etc.

(2) Grassmann_Optimization.m

the object class for optimization calculus and differential geometry on Grassmann manifolds, such as tangent projection, exponential map, geodesics, gradient descent, retraction, lifting, logarithmic map, etc.

(3) buildVisualWordList.m

partition a given sample data set according to a tree of given height into leaf nodes

(4) SIFT_PCA.m

do the SIFT (Scale Invariant Feature Transform) PCA analysis

(5) SIFT_PCA_Recovery.m

do the SIFT PCA recovery using the Stiefel_Optimization method, compare with benchmark nearest neighbor method

(6) LPP_CenterMass.m

classfication analysis based on Laplacian eigenface and graph Laplacian method, as well as center of mass on Grassmann manifold. Applied to several different datasets: nwpu-aerial-images, MNIST, cifar10

<b>(b) folder "python_code"</b>

(7) Stiefel_Optimization.py 

file with same name and function as matlab_code

(8) Grassmann_Optimization.py

file with same name and function as matlab_code

(9) buildVisualWordList.py

file with same name and function as matlab_code

(10) LPP_CenterMass.py

classfication analysis based on Laplacian eigenface and graph Laplacian method, as well as center of mass on Grassmann manifold. Applied to several different datasets: MNIST, cifar10; incorporates GMM sampling of pseudo-data inputs and labeling by pre-trained model

(11) cifar10vgg.py

build a pre-trained vgg model for cifar10, can also train a new cifar10. Pre-trained model paramter data available at https://github.com/geifmany/cifar-vgg
