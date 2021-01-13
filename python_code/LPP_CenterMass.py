"""
Created on Mon Aug  3 17:32:08 2020

%%%%%%%%%%%%%%%%%%%% LPP analysis based on Grassmann center of mass calculation %%%%%%%%%%%%%%%%%%%%

@author: Wenqing Hu (Missouri S&T)
"""

from Stiefel_Optimization import Stiefel_Optimization
from Grassmann_Optimization import Grassmann_Optimization
from buildVisualWordList import buildVisualWordList
from umap_data_aug import UMAP_Augmentation
from operator import itemgetter
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import tensorflow as tf
import time
from cifar10vgg import cifar10vgg
from sklearn.mixture import GaussianMixture


# set the pre-trained learning models
model_cifar10vgg = cifar10vgg()


# load the data set, MNIST or CIFAR-10
def load_data(doMNIST, doCIFAR10):
    # load MNIST dataset
    if doMNIST:
        # load the MNIST dataset
        # structure: 
        #    x_train: list (60000, 28 , 28) dtype=unit8
        #    x_test: list (10000, 28 , 28) dtype=unit8
        #    y_train: list (60000, 1) dtype=unit8
        #    y_test: list (10000, 1) dtype=unit8
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # preprocess the dataset to fit the format we use
        data_original_train = {"x": [], "y": []}
        data_original_test = {"x": [], "y": []}
        # first turn the matrices of x_train and x_test to 28 x 28 = 784 dimensional vectors
        for i in range(60000):
            data_original_train["x"].append(np.reshape(x_train[i], 784))
            data_original_train["y"].append(y_train[i])
        for i in range(10000):
            data_original_test["x"].append(np.reshape(x_test[i], 784))
            data_original_test["y"].append(y_test[i])
    # load CIFAR10 dataset
    if doCIFAR10:
        # load the CIFAR-10 dataset
        # structure: 
        # structure: 
        #    x_train: list (50000, 32 , 32, 3) dtype=unit8
        #    x_test: list (10000, 32 , 32, 3) dtype=unit8
        #    y_train: list (50000, 1) dtype=unit8
        #    y_test: list (10000, 1) dtype=unit8
        cifar10 = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # preprocess the dataset to fit the format we use
        data_original_train = {"x": [], "y": []}
        data_original_test = {"x": [], "y": []}
        # first turn the matrices of x_train and x_test to 32 x 32 x 3 = 3072 dimensional vectors
        for i in range(50000):
            data_original_train["x"].append(np.reshape(x_train[i], 3072))
            data_original_train["y"].append(y_train[i][0])
        for i in range(10000):
            data_original_test["x"].append(np.reshape(x_test[i], 3072))
            data_original_test["y"].append(y_test[i][0])
            
    return data_original_train, data_original_test


# k-nearest neighbor classfication
# given test data x and label y, find in a training set (X, Y) the k-nearest points x1,...,xk to x, and classify x as majority vote on y1,...,yk
# if the classification is correct, return 1, otherwise return 0
def knn(x_test, y_test, X_train, Y_train, k):
    m = len(Y_train)
    if k>m:
        k=m
    # find the first k-nearest neighbor
    dist = [np.linalg.norm(np.array(x_test)-np.array(X_train[i])) for i in range(m)]
    indexes, dist_sort = zip(*sorted(enumerate(dist), key=itemgetter(1))) 
    # do a majority vote on the first k-nearest neighbor
    label = [Y_train[indexes[_]] for _ in range(k)]
    vote = pd.value_counts(label)
    # class_predict is the predicted label based on majority vote
    class_predict = vote.index[0]
    if class_predict == y_test:
        isclassified = 1
    else:
        isclassified = 0
    return isclassified, class_predict


# solve the laplacian embedding, given data set X={x1,...,xm}, the graph laplacian L and degree matrix D    
def LPP(X, L, D):
    # turn X, L, D into arrays
    X = np.array(X)
    L = np.array(L)
    D = np.array(D)
    # calculate mtx_L = X' * L * X
    mtx_L = np.matmul(np.matmul(X.T, L), X)
    # calculate mtx_D = X' * D * X
    mtx_D = np.matmul(np.matmul(X.T, D), X)
    # solve the generalized eigenvalue problem mtx_L W = LAMBDA mtx_D W
    LAMBDA, W = eigh(mtx_L, mtx_D, eigvals_only=False)
    # sort the eigenvalues in a descending order
    SORT_ORDER, LAMBDA = zip(*sorted(enumerate(LAMBDA), key=itemgetter(1), reverse=False)) 
    # reorder the generalized eigenvector matrix W according to SORT_ORDER
    W = [W[SORT_ORDER[_]] for _ in range(len(SORT_ORDER))]
    return W, LAMBDA 
    
 
# construct the graph laplacian L and the degress matrix D from the given affinity matrix S 
def graph_laplacian(S):
    # first turn S into an array
    S = np.array(S)
    # compute the D matrix
    D = np.diag(sum(S, 0))
    L = D - S
    return L, D


# given a set of data points X={x1,...,xm} with label Y={y1,...,ym}, construct their supervised affinity matrix S for LPP
def affinity_supervised(X, Y, between_class_affinity):
    # original distances squares between xi and xj
    f_dist1 = cdist(X, X, 'euclidean')
    # heat kernel size
    mdist = np.mean(f_dist1) 
    h = -np.log(0.15)/mdist
    S1 = np.exp(-h*f_dist1)
    # utilize supervised info
    # first turn Y into a 2-d array
    Y = [[Y[_]] for _ in range(len(Y))]
    id_dist = cdist(Y, Y, 'euclidean')
    S2 = S1 
    for i in range(len(X)):
        for j in range(len(X)):
            if id_dist[i][j] != 0:
                S2[i][j] = between_class_affinity
    # obtain the supervised affinity S
    S = S2
    return S


# Sample a training dataset data_train from the data_original_train set, data_train = (data_train["x"], data_train["y"])
# Set the partition tree depth = ht
# Tree partition data_train into clusters C_1, ..., C_{2^{ht}} with centers m_1, ..., m_{2^{ht}}
# Sample a test dataset data_test from the data_original_test set for testing purposes, data_test = (data_test["x"], data_test["y"])
def LPP_ObtainData(data_original_train, data_original_test, d_PCA, d_SecondPCA_kdtree, train_size, test_size, ht):
    # Input
    #   data = the original data set, in the python code we treat it as a dictionary data={"x": [inputs], "y": [labels]}
    #   d_PCA = the data preprocessing PCA projection dimension
    #   d_SecondPCA_kdtree = the second level PCA embedding dimension for building the kd-tree
    #   train_size, test_size = the training/testing data set size
    #   ht = the partition tree height
    # Output
    #   data_train, data_test = the training/testing data set , size is traing_size/test_size
    #   leafs = leafs{k}, the cluster indexes in data_train
    #   inv_mat = the pseudo-inverse map that helps to reconstruct the labels for newly-generated training data x using pre-trained model
    
    # compute the sizes of the original training and testing dataset
    n_data_original_train = len(data_original_train["x"]) 
    n_data_original_test = len(data_original_test["x"])
    
    # choose to do preliminary dimension reduction for computational feasability only
    if do_preliminary_PCA_reduction:
        # first concatnate data_original_train["x"] and data_original_test["x"] together
        data_x = np.array(data_original_train["x"] + data_original_test["x"])
        # do an initial PCA on data
        pca = PCA()
        pca.fit(data_x)
        A0 = pca.components_
        # bulid a given dimensional d_PCA embedding of data_orginal_train(test).x into new data_original_train(test).x, for faster computation only
        data_original_train["x"] = np.matmul(data_original_train["x"], np.array([A0[_] for _ in range(d_PCA)]).T)
        data_original_test["x"] = np.matmul(data_original_test["x"], np.array([A0[_] for _ in range(d_PCA)]).T)
        # record the pseudo-inverse map that helps to recover the low-dimensional data to original data dimension
        inv_mat = np.linalg.pinv(np.array([A0[_] for _ in range(d_PCA)]).T)
    else:
        # record the pseudo-inverse map that helps to recover the low-dimensional data to original data dimension
        d_data = len(data_original_train["x"][0])
        inv_mat = np.identity(d_data, dtype=float)
    
    # build the training data set
    indexes = np.random.permutation(n_data_original_train) 
    # randomly pick the training sample of size train_size from data_original_train dataset
    train_indexes = [indexes[_] for _ in range(train_size)]
    # form the data_train dataset
    data_train_x = [data_original_train["x"][_] for _ in train_indexes]
    data_train_y = [data_original_train["y"][_] for _ in train_indexes]
    data_train = {"x": data_train_x, "y": data_train_y}
    
    # choose to augment the obtained training data x and y globally using GMM sampling and label the new x inputs using pre-trained learning model
    if doAugment_Global:
        # bulid a Gaussian mixture model on data_train_x
        # sample from this GMM enough number of training data points, form data_train_x
        # use the pre-trained learning model, predict labels for the newly generated training set
        data_train_x_additional, data_train_y_additional = TrainingDataAugmentation(data_train_x,
                                                                                    data_train_y, 
                                                                                    number_samples_additional_Global,
                                                                                    number_components_Global,
                                                                                    learning_model,
                                                                                    inv_mat)
        data_train_x.extend(data_train_x_additional)
        data_train_y.extend(data_train_y_additional)        
    
    # build the testing data set
    indexes = np.random.permutation(n_data_original_test) 
    # randomly pick the test sample of size test_size from data_original_test dataset
    test_indexes = [indexes[_]  for _ in range(test_size)]
    # form the data_test dataset
    data_test_x = [data_original_test["x"][_] for _ in test_indexes]
    data_test_y = [data_original_test["y"][_] for _ in test_indexes]
    data_test = {"x": data_test_x, "y": data_test_y}
    
    # choose to do a second level PCA to dimension d_SecondPCA_kdtree before the kd-tree decomposition
    if doSecondPCA_kdtree:
        # do another initial PCA on data_train to d_SecondPCA_kdtree
        pca = PCA()
        pca.fit(data_train_x)
        A0 = pca.components_
        # bulid a d_SecondPCA_kdtree dimensional embedding of data_train in x0
        x0 = np.matmul(data_train_x, np.array([A0[_] for _ in range(d_SecondPCA_kdtree)]).T)
        # from x0, partition into 2^ht leaf nodes, each leaf node can give samples for a local LPP
        indx, leafs, mbrs = buildVisualWordList(x0, ht)
    else:
        # from data_train_x, partition into 2^ht leaf nodes, each leaf node can give samples for a local LPP
        indx, leafs, mbrs = buildVisualWordList(data_train_x, ht)
    
    return data_train, leafs, data_test, inv_mat


# build LPP Model for each leaf in data_train
# Assume the tree partition indexes of data_train into clusters C_1, ..., C_{2^{ht}} with centers m_1, ..., m_{2^{ht}} is given in leafs
# first project each C_i to local PCA with dimension d_SecondPCA_beforeLPP  
# then continue to construct the local LPP frames A_1, ..., A_{2^{ht}} in G(d_data, d_LPP) using supervised affinity
def LPP_BuildDataModel(data_train, leafs, d_SecondPCA_beforeLPP, d_LPP, inv_mat, train_size):
    # Input:
    #   data_train = the training data set
    #   leafs = the tree partition indexes of data_train into clusters C_1, ..., C_{2^{ht}}
    #   d_SecondPCA_beforeLPP = the second-level PCA embedding dimension before we do LPP
    #   d_LPP = the LPP embedding dimension 
    #   inv_mat = the pseudo-inverse map that helps to reconstruct the labels for newly-generated training data x within kd-tree cluster using pre-trained model
    #   train_size = the size of the original training data set
    # Output:
    #   Seq = the LPP frames corresponding to each cluster in data_train, labeling the correponding Grassmann equivalence class
    #   data_train = the training data set possibly modified by augmenting each cluster using pre-trained model labeling
    #   leafs = the indexes of data_train into new possibly augmented clusters C_1, ..., C_{2^{ht}}

    # obtain the dimension of each sample in data_train["x"]
    d_data = len(data_train["x"][0])
    # initialize the LPP frames A_1,...,A_{2^{ht}}
    Seq = np.zeros((len(leafs), d_data, d_LPP))
    # build LPP Model for each leaf
    # input: data, indx, leafs
    for k in range(len(leafs)):
        # form the data_train subsample the k-th cluster
        data_train_x_k = [data_train["x"][_] for _ in leafs[k]]
        data_train_y_k = [data_train["y"][_] for _ in leafs[k]]
        # augment the data_train_x_k and data_train_y_k by GMM sampling and pre-trained learning model prediction
        if doAugment_kdtreeCluster:
            # bulid a Gaussian mixture model on data_train_x_k
            # sample from this GMM enough number of training data points, form data_train_x_k
            # use the pre-trained learning model, predict labels for the newly generated training set
            data_train_x_k_additional, data_train_y_k_additional = TrainingDataAugmentation(data_train_x_k, 
                                                                                            data_train_y_k, 
                                                                                            number_samples_additional_kdtreeCluster,
                                                                                            number_components_kdtreeCluster,
                                                                                            learning_model,
                                                                                            inv_mat)
            data_train_x_k.extend(data_train_x_k_additional)
            data_train_y_k.extend(data_train_y_k_additional)

        if doSecondPCA_beforeLPP:
            # do a second-level PCA first, for the k-th cluster, so data_train_x_k dimension is reduced to d_SecondPCA_beforeLPP
            pca = PCA()
            pca.fit(data_train_x_k)
            PCA_k = pca.components_
            data_train_x_k = np.matmul(data_train_x_k, np.array([PCA_k[_] for _ in range(d_SecondPCA_beforeLPP)]).T)
            # then do LPP for the PCA embedded data_train_x_k and reduce the dimension to d_LPP
            # construct the supervise affinity matrix S
            between_class_affinity = 0
            S_k = affinity_supervised(data_train_x_k, data_train_y_k, between_class_affinity)
            # construct the graph Laplacian L and degree matrix D
            L_k, D_k = graph_laplacian(S_k)
            # do LPP
            A_k, LAMBDA = LPP(data_train_x_k, L_k, D_k)
            LPP_k, R = np.linalg.qr(A_k)        
            # obtain the frame Seq(:,:,k)
            Seq[k] = np.matmul(np.array([PCA_k[_] for _ in range(d_SecondPCA_beforeLPP)]).T, np.array([LPP_k[_] for _ in range(1, d_LPP+1)]).T)
            print("frame ",k+1," size=(", len(Seq[k]),",",len(Seq[k][0]), "), IfStiefel? Residue = ", np.linalg.norm(np.array(np.matmul(Seq[k].T, Seq[k]))-np.array(np.diag(np.ones(d_LPP)))))
        else:
            # do LPP directly to data_train_x_k and reduce the dimension to d_LPP
            # construct the supervise affinity matrix S
            between_class_affinity = 0
            S_k = affinity_supervised(data_train_x_k, data_train_y_k, between_class_affinity)
            # construct the graph Laplacian L and degree matrix D
            L_k, D_k = graph_laplacian(S_k)
            # do LPP
            A_k, LAMBDA = LPP(data_train_x_k, L_k, D_k)
            LPP_k, R = np.linalg.qr(A_k)        
            # obtain the frame Seq(:,:,k)
            Seq[k] = np.array([LPP_k[_] for _ in range(1, d_LPP+1)]).T
            print("frame ",k+1," size=(", len(Seq[k]),",",len(Seq[k][0]), "), IfStiefel? Residue = ", np.linalg.norm(np.array(np.matmul(Seq[k].T, Seq[k]))-np.array(np.diag(np.ones(d_LPP)))))

    # choose to use the augmented data with labels from pre-trained model for the clusters
    if doUseAugmentData_kdtreeCluster and doAugment_kdtreeCluster:
        for k in range(len(leafs)):
            # finalize the training data for the k-th cluster
            data_train["x"].extend(data_train_x_k_additional)    
            data_train["y"].extend(data_train_y_k_additional)
            leafs[k].extend(range(train_size + k * number_samples_additional_kdtreeCluster, train_size + (k+1) * number_samples_additional_kdtreeCluster))
   
    return Seq, data_train, leafs


# Test the LPP piecewise linear embedding model and the interpolated piecewise linear embedding model
# Using k-nearest neighbor classification method
# Also compare with the nearest neighbor classification in the original dimension
def LPP_NearestNeighborTest():

    # load data
    data_original_train, data_original_test = load_data(doMNIST, doCIFAR10)

    # obtain the train, test sets in nwpu and the LPP frames Seq(:,:,k) for each cluster with indexes in leafs
    data_train, leafs, data_test, inv_mat = LPP_ObtainData(data_original_train, data_original_test, d_PCA, d_SecondPCA_kdtree, train_size, test_size, ht)
    Seq, data_train, leafs = LPP_BuildDataModel(data_train, leafs, d_SecondPCA_beforeLPP, d_LPP, inv_mat, train_size)

    # all these LPP Stiefel frames are on St(n, p)
    n = len(Seq[0])
    p = len(Seq[0][0])
        
    # data original dimension d_data
    d_data = len(data_train["x"][0])
        
    # find m_1, ..., m_{2^{ht}}, the means of the chosen clusters
    m = np.zeros((2**ht, d_data))
    for k in range(2**ht):
        m[k] = np.mean([data_train["x"][_] for _ in leafs[k]], axis=0)

    # set the sequence of interpolation numbers and the threshold ratio for determining the interpolation number
    interpolation_number_seq = np.ones(test_size)
    ratio_seq = np.zeros((test_size, 2)) # the sequence of second smallest (or largest) to-center distance over smallest to-center distance, for tuning ratio_threshold
   
    classified_o = np.zeros(test_size) # list of classified/not classified projections for using the original data point and nearest cluster
    classified_agg_o = np.zeros(test_size) # list of classified/not classified projections for using the original data point and nearest (interpolation_number) clusters
    classified_bm = np.zeros(test_size) # list of classified/not classified projections for using the nearest frame, benchmark
    classified_c = np.zeros(test_size) # list of classified/not classified projections for using the Grassmann center method
    classified_model = np.zeros(test_size) # list of classified/not classified projections for using the pre-trained learning model
    
    cpu_time_start = time.process_time()
    for test_index in range(test_size):
        print("\ntest point", test_index+1, " -----------------------------------------------------------\n")
        x = data_test["x"][test_index]
        y = data_test["y"][test_index]
        # sort the cluster centers m_1, ..., m_{2^{ht}} by ascending distances to x 
        dist = [np.linalg.norm(x-m[k]) for k in range(2**ht)]
        indexes, dist_sort = zip(*sorted(enumerate(dist), key=itemgetter(1))) 
        # count the number of St(p, n) interpolation clusters for current test point x
        # interpolation_number = number of frames used for interpolation between cluster LDA frames
        interpolation_number = 1
        print("ratio between [", dist_sort[1]/dist_sort[0], ",", dist_sort[2**ht-1]/dist_sort[0], "]")
        ratio_seq[test_index][0] = dist_sort[1]/dist_sort[0]
        ratio_seq[test_index][1] = dist_sort[2**ht-1]/dist_sort[0]
        for k in range(1, 2**ht):
            if dist_sort[k] <= ratio_threshold * dist_sort[0]:
                interpolation_number = interpolation_number + 1
            else:
                break
        print("interpolation number = ", interpolation_number)
        # record the sequence of all interpolation numbers for each test point x
        interpolation_number_seq[test_index] = interpolation_number
        # find the LPP Stiefel projection frames A_k1, ..., A_k{interpolation_number} for the first (interpolation_number) closest clusters to x
        frames = np.zeros((interpolation_number, d_data, d_LPP))
        for i in range(interpolation_number):
            frames[i] = Seq[indexes[i]]
        # find the weights w_1, ..., w_{interpolation_number} for the first (interpolation_number) closest clusters to x
        w = [np.exp(-K * (dist_sort[i]**2)) for i in range(interpolation_number)]
        # collect all indexes in clusters corresponding to the first (interpolation_number) closest clusters to x
        aggregate_cluster = []
        for i in range(interpolation_number):
            aggregate_cluster = list(set(aggregate_cluster) | set(leafs[indexes[i]]))
        # do k-nearest-neighbor classification based on the closest cluster to x, in original space
        x_test = x
        y_test = y
        X_train = [data_train["x"][_] for _ in leafs[indexes[0]]]
        Y_train = [data_train["y"][_] for _ in leafs[indexes[0]]]
        isclassified_o, class_predict = knn(x_test, y_test, X_train, Y_train, k_nearest_neighbor)
        classified_o[test_index] = isclassified_o
        # do k-nearest-neighbor classification based on the (interpolation_number) nearest clusters to x, in oroginal space
        x_test = x
        y_test = y
        X_train = [data_train["x"][_] for _ in aggregate_cluster]
        Y_train = [data_train["y"][_] for _ in aggregate_cluster]
        isclassified_agg_o, class_predict = knn(x_test, y_test, X_train, Y_train, k_nearest_neighbor)
        classified_agg_o[test_index] = isclassified_agg_o
        # project x to A1 x and classify it using k-nearest-neighbor on the projection via A1 of the closest cluster
        x_test = np.matmul(x, frames[0])
        y_test = y
        X_train = [np.matmul(data_train["x"][_],  frames[0]) for _ in leafs[indexes[0]]]
        Y_train = [data_train["y"][_] for _ in leafs[indexes[0]]]
        isclassified_bm, class_predict = knn(x_test, y_test, X_train, Y_train, k_nearest_neighbor)
        classified_bm[test_index] = isclassified_bm
        # calculate the center of mass for the (interpolation_number) nearest cluster LPP frames with respect to weights w 
        if doGrassmannpFCenter:
            # do Grassmann center of mass method
            GrassmannOpt = Grassmann_Optimization(w, frames, threshold_gradnorm, threshold_fixedpoint, threshold_checkonGrassmann)
            if doGD:
                break
            else:
                center, value, grad = GrassmannOpt.Center_Mass_pFrobenius()
        else:
            # do Stiefel center of mass method
            StiefelOpt = Stiefel_Optimization(w, frames, threshold_gradnorm, threshold_fixedpoint, threshold_checkonStiefel, threshold_logStiefel)
            if doStiefelEuclidCenter:
                if doGD:
                    break
                else:
                    center, value, gradnorm = StiefelOpt.Center_Mass_Euclid()
            else:
                break
        # project x to center x and classify it using k-nearest-neighbor on the projection via center of all (interpolation number) clusters
        x_test = np.matmul(x , center)
        y_test = y
        X_train = [np.matmul(data_train["x"][_], center) for _ in aggregate_cluster]
        Y_train = [data_train["y"][_] for _ in aggregate_cluster]    
        isclassified_c, class_predict = knn(x_test, y_test, X_train, Y_train, k_nearest_neighbor)
        classified_c[test_index] = isclassified_c
        # classify x using pre-trained learning model
        x_test = x
        y_test = y
        if learning_model == 'cifar10vgg':
            model = model_cifar10vgg
            x_test_ = [x_test]
            x_test__ = np.matmul(x_test_, inv_mat)
            x_test___ = np.reshape(x_test__.flatten(), (1, 32, 32, 3))
            predicted_x = model.predict(x_test___)
            class_predict = np.argmax(predicted_x, 1)[0]
            isclassified_model = (class_predict == y) + 0
        else: 
            print("No Pre-Trained Learning Model Chosen!\n")
            isclassified_model = 0
        classified_model[test_index] = isclassified_model
        
        # output the result
        print("original dimension classified =", isclassified_o)
        print("original dimension aggregate classified =", isclassified_agg_o)
        print("benchmark classified =", isclassified_bm)
        print("center mass classfied =", isclassified_c)
        print("pre-trained model clssified =", isclassified_model)

    # summarize the final result
    cpu_time_end = time.process_time()
    cpu_time = cpu_time_end - cpu_time_start
    rate_o = (sum(classified_o)/test_size)*100
    rate_agg_o = (sum(classified_agg_o)/test_size)*100
    rate_bm = (sum(classified_bm)/test_size)*100
    rate_c = (sum(classified_c)/test_size)*100
    rate_model = (sum(classified_model)/test_size)*100
    print("\n******************** CONCLUSION ********************")
    print("\ncpu runtime for testing = ", cpu_time_end - cpu_time_start, " seconds \n")
    print("\nClassification rates\n")
    print("\nOption 1. using the original data point and nearest cluster: ", rate_o, "%")
    print("\nOption 2. using the original data point and nearest (interpolation_number) clusters:", rate_agg_o, "%")
    print("\nOption 3. using the nearest cluster LPP frame after LPP projection, benchmark: ", rate_bm, "%")
    print("\nOption 4. using the Grassmann center obtained from several nearest cluster LPP frames after LPP projection:", rate_c, "%")
    print("\nOption 5. using the pre-trained learning model and the pseudo-invese of the initial PCA =", rate_model, "%\n")

    file=open('conclusion.txt', 'w')
    print("\n******************** CONCLUSION ********************", file=file)
    print("\ncpu runtime for testing = ", cpu_time_end - cpu_time_start, " seconds \n", file=file)
    print("\nClassification rates\n", file=file)
    print("\nOption 1. using the original data point and nearest cluster: ", rate_o, "%", file=file)
    print("\nOption 2. using the original data point and nearest (interpolation_number) clusters:", rate_agg_o, "%", file=file)
    print("\nOption 3. using the nearest cluster LPP frame after LPP projection, benchmark: ", rate_bm, "%", file=file)
    print("\nOption 4. using the Grassmann center obtained from several nearest cluster LPP frames after LPP projection:", rate_c, "%", file=file)
    print("\nOption 5. using the pre-trained learning model and the pseudo-invese of the initial PCA =", rate_model, "%\n", file=file)
    file.close()

    return cpu_time, rate_o, rate_agg_o, rate_bm, rate_c, rate_model


# given a set of training_data_original_x with labels training_data_original_y
# generate a given number of additional training samples training_data_additional_x 
# with training_data_additional_x, using a pre-trained learning_model, label each additional sample and produce corresponding labels training_data_additional_y
def TrainingDataAugmentation(training_data_original_x, training_data_original_y, number_samples_additional, number_components, learning_model, inv_mat):

    if doAugmentViaGMM:
        # fit train_data_original_x using a GMM model
        gmm = GaussianMixture(n_components = number_components).fit(training_data_original_x)
        # using GMM, generate an additional set of training_data_additional_x and predict training_data_additional_y
        training_data_additional_x_, y = gmm.sample(number_samples_additional)
    elif doAugmentViaUMAP:
        # augment train_data_original_x using UMAP
        training_data_additional_x_ = UMAP_Augmentation(training_data_original_x, training_data_original_y, number_components, number_samples_additional, number_neighbors_UMAP)
    else:
        # do nothing
        print("No Data Augmentation Method Chosen!\n")
        return None

    # initialize the new labels
    training_data_additional_y = []
    
    if learning_model == 'cifar10vgg':
        model = model_cifar10vgg
        for i in range(number_samples_additional):
            training_data_additional_x__i = np.matmul(training_data_additional_x_[i], inv_mat)
            training_data_additional_x___i = np.reshape(training_data_additional_x__i.flatten(), (1, 32, 32, 3))
            predicted_x_i = model.predict(training_data_additional_x___i)
            training_data_additional_y.append(np.argmax(predicted_x_i, 1)[0])
            print("Newly generated input data #", i, ", pre-trained model predicted label is ", training_data_additional_y[i])
    elif learning_model == 'GMM':
        training_data_additional_y = (gmm.predict(training_data_additional_x_)).tolist()
    else:
        print("No Pre-Trained Learning Model Chosen!\n")
        return None

    training_data_additional_x = [np.array(training_data_additional_x_[_]) for _ in range(number_samples_additional)]
     
    return training_data_additional_x, training_data_additional_y



"""
################################ MAIN RUNNING FILE #####################################

LPP analysis based on Grassmann center of mass calculation
"""

if __name__ == "__main__":

    ###############################################################################################################
    ########################### set all parameters needed in the code for easier tuning ###########################
    ###############################################################################################################

    # choose to do preliminary PCA dimension reduction to d_PCA for computational feasability only
    do_preliminary_PCA_reduction = 1
    # choose to do another PCA to dimension d_SecondPCA_kdtree before do kd-tree decomposition
    doSecondPCA_kdtree = 0
    # choose to do a PCA for each cluster to dimension d_SecondPCA_beforeLPP before do LPP on that cluster
    doSecondPCA_beforeLPP = 0

    # select which dataset to work on
    doMNIST = 0
    doCIFAR10 = 1
    # the data preprocessing preliminary PCA reduction projection dimension
    d_PCA = 256
    # the secondary PCA embedding dimension in case we do a second PCA to dimension d_SecondPCA_beforeLPP before the kd-tree decomposition into clusters
    d_SecondPCA_kdtree = 128
    # the secondary PCA embedding dimension in case we do a second PCA for each cluster to dimension d_SecondPCA_kdtree before we do LPP on that cluster
    d_SecondPCA_beforeLPP = 100
    # the LPP embedding dimension = d_LPP on each given cluster
    d_LPP = 128
    # train_size = the training data size
    train_size = 150 * (2**8)
    # ht = the partition tree height
    ht = 8
    # test_size = the test data size
    test_size = 1000


    # choose to augment the original training data x and y globally by GMM sampling and pre-trained learning model prediction, use them to build the kd-tree and subspace model
    # in this case, the augmented data points will be used automatically in knn nearest neighbor clssification
    doAugment_Global = 0
    # the number of additional samples for the whole training set, in case we do augment the training set globally
    number_samples_additional_Global = 200 * (2**8)
    # the number of components used in gmm when generating new training data x globally for the whole training set, it is different from label y classes in the training data 
    number_components_Global = 2
    # choose to augment the data_train_x_k and data_train_y_k within the kd tree cluster by GMM sampling and pre-trained learning model prediction, use them to build the subspace model
    doAugment_kdtreeCluster = 1
    # choose to use the augmented data developed for each kd tree cluster in doing nearest neighbor classification
    doUseAugmentData_kdtreeCluster = 1
    # the number of additional samples in a kd-tree cluster, in case we do augment training data within that kd-tree cluster
    number_samples_additional_kdtreeCluster = 500
    # the number of components used in gmm when generating new training data x within a kd-tree cluster, it is different from label y classes in the training data 
    number_components_kdtreeCluster = 2
    # pick the method of augmentation: GMM, UMAP
    doAugmentViaGMM = 0
    doAugmentViaUMAP = 1
    # parameters for UMAP
    number_neighbors_UMAP = 20
    # pick the pre-trained learning model for augmentation
    doCIFAR10vgg = 1
    doGMM = 0
    if doCIFAR10vgg:
        learning_model = 'cifar10vgg'  
    elif doGMM:
        learning_model = 'GMM' 
    else:
        learning_model = 'NoModel'

    # the ratio for determinining the interpolation_number, serve as a tuning parameter
    ratio_threshold = 1.2 
    # the scaling coefficient for calculating the weights w = e^{-K distance^2}
    K = 1e-8 
    # the parameter k for k-nearest-neighbor classification
    k_nearest_neighbor = 1 
    # do or do not do projected Frobenius center of mass for Grassmannian frame    
    doGrassmannpFCenter = 0 
    # do or do not do Euclid center of mass for Stiefel frame     
    doStiefelEuclidCenter = 1 
    # do or do not do GD for finding center of mass     
    doGD = 0 
    # threshold parameters for Stiefel and Grassmann Optimization
    threshold_gradnorm = 1e-4
    threshold_fixedpoint = 1e-4
    threshold_checkonGrassmann = 1e-10
    threshold_checkonStiefel = 1e-10
    threshold_logStiefel = 1e-4

    # do test correctness of the specific functions developed
    doRunTest=0
    # do the test of the classification rate using original full data set and original dimension
    # can choose the data set to be augmented by the pre-trained model, either globally or by each cluster 
    doTestFullData_knn = 0
    # do the LPP analysis on different datasets
    doLPP_NearestNeighborTest = 1

    ###############################################################################################################
    ###########################                 end of parameter setting                ###########################
    ###############################################################################################################

    # do test correctness of the specific functions developed
    if doRunTest:
        x = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30], [31, 32]]
        ht = 2
        indx, leafs, mbrs = buildVisualWordList(x, ht)
        print("leafs=", leafs)
        print("indx=", indx)
        print("mbrs=", mbrs)
        
        x_test = [0, 0]
        y_test = 2
        X_train = [[0, 1], [1, 0], [0, 2], [2, 0], [0, 3], [3, 0]]
        Y_train = [2, 2, 2, 2, 1, 1]
        k = 6
        isclassified = knn(x_test, y_test, X_train, Y_train, k)
        print("isclassified=", isclassified)
        
        S = [[2, 1], [1, 2]]
        L, D = graph_laplacian(S)
        print("L=", L, "D=", D)
        X = np.array([[0, 1], [1, 0]])
        W, LAMBDA = LPP(X, L, D)
        print("W=", W)
        print("LAMBDA=", LAMBDA)
        
        X = [[0, 1, 2], [2, 3, 4], [4, 5, 6]]
        Y = [1, 2, 1]
        between_class_affinity = 0
        S = affinity_supervised(X, Y, between_class_affinity)
        print("S=", S)
        
    # do the test of the classification rate using original full data set and original dimension
    # can choose the data set to be augmented by the pre-trained model, either globally or by each cluster 
    if doTestFullData_knn:
        # load data
        data_original_train, data_original_test = load_data(doMNIST, doCIFAR10)
        # obtain the train, test sets in nwpu and the LPP frames Seq(:,:,k) for each cluster with indexes in leafs
        data_train, leafs, data_test, inv_mat = LPP_ObtainData(data_original_train, data_original_test, d_PCA, d_SecondPCA_kdtree, train_size, test_size, ht)
        if doUseAugmentData_kdtreeCluster:
            Seq, data_train, leafs = LPP_BuildDataModel(data_train, leafs, d_SecondPCA_beforeLPP, d_LPP, inv_mat, train_size)
        classified_fulldataset = np.zeros(test_size) # list of classified/not classified projections for using knn in the whole data set in itr original space
        for test_index in range(test_size):
            print("\ntest point", test_index+1, " -----------------------------------------------------------\n")
            x = data_test["x"][test_index]
            y = data_test["y"][test_index]
            # do k-nearest-neighbor classification for all training data in the original space
            x_test = x
            y_test = y
            X_train = data_train["x"]
            Y_train = data_train["y"]
            isclassified_fulldataset, class_predict = knn(x_test, y_test, X_train, Y_train, k_nearest_neighbor)
            classified_fulldataset[test_index] = isclassified_fulldataset
            print("full dataset in original dimension classified =", isclassified_fulldataset)
        # summarize the final result
        print("\nfull data set original dimension classification rate = ", (sum(classified_fulldataset)/test_size)*100, "%")

    # do the LPP analysis on different datasets
    if doLPP_NearestNeighborTest:
        cpu_time, rate_o, rate_agg_o, rate_bm, rate_c, rate_model = LPP_NearestNeighborTest()

