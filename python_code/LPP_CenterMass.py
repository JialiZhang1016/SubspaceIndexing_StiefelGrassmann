#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 17:32:08 2020

%%%%%%%%%%%%%%%%%%%% LPP analysis based on Grassmann center of mass calculation %%%%%%%%%%%%%%%%%%%%

@author: Wenqing Hu (Missouri S&T)
"""

from Stiefel_Optimization import Stiefel_Optimization
from buildVisualWordList import buildVisualWordList
from operator import itemgetter
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import tensorflow as tf
import time
from cifar10vgg import cifar10vgg


# load the data set, nwpu-aerial-images, MNIST or CIFAR-10
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
def LPP_ObtainData(data_original_train, data_original_test, d_pre, kd_LPP, kd_PCA, train_size, test_size, ht):
    # Input
    #   data = the original data set, in the python code we treat it as a dictionary data={"x": [inputs], "y": [labels]}
    #   d_pre = the data preprocessing projection dimension
    #   kd_PCA = the initial PCA embedding dimension
    #   kd_LPP = the LPP embedding dimension 
    #   train_size, test_size = the training/testing data set size
    #   ht = the partition tree height
    # Output
    #   data_train, data_test = the training/testing data set , size is traing_size/test_size
    #   leafs = leafs{k}, the cluster indexes in data_train
    
    # compute the sizes of the original training and testing dataset
    n_data_original_train = len(data_original_train["x"]) 
    n_data_original_test = len(data_original_test["x"])
    
    # choose to do preliminary dimension reduction for computational feasability only
    do_preliminary_reduction = 0
    if do_preliminary_reduction:
        # first concatnate data_original_train["x"] and data_original_test["x"] together
        data_x = np.array(data_original_train["x"] + data_original_test["x"])
        # do an initial PCA on data
        pca = PCA()
        pca.fit(data_x)
        A0 = pca.components_
        # bulid a given dimensional d_pre embedding of data_orginal_train(test).x into new data_original_train(test).x, for faster computation only
        data_original_train["x"] = np.matmul(data_original_train["x"], np.array([A0[_] for _ in range(d_pre)]).T)
        data_original_test["x"] = np.matmul(data_original_test["x"], np.array([A0[_] for _ in range(d_pre)]).T)
    
    # build the training data set
    indexes = np.random.permutation(n_data_original_train) 
    # randomly pick the training sample of size train_size from data_original_train dataset
    train_indexes = [indexes[_] for _ in range(train_size)]
    # form the data_train dataset
    data_train_x = [data_original_train["x"][_] for _ in train_indexes]
    data_train_y = [data_original_train["y"][_] for _ in train_indexes]
    data_train = {"x": data_train_x, "y": data_train_y}
    
    # build the testing data set
    indexes = np.random.permutation(n_data_original_test) 
    # randomly pick the test sample of size test_size from data_original_test dataset
    test_indexes = [indexes[_]  for _ in range(test_size)]
    # form the data_test dataset
    data_test_x = [data_original_test["x"][_] for _ in test_indexes]
    data_test_y = [data_original_test["y"][_] for _ in test_indexes]
    data_test = {"x": data_test_x, "y": data_test_y}
    
    # do an initial PCA on data_train
    pca = PCA()
    pca.fit(data_train_x)
    A0 = pca.components_
    # bulid a kd_PCA dimensional embedding of data_train in x0
    x0 = np.matmul(data_train_x, np.array([A0[_] for _ in range(kd_PCA)]).T)
    # from x0, partition into 2^ht leaf nodes, each leaf node can give samples for a local LPP
    indx, leafs, mbrs = buildVisualWordList(x0, ht)
    
    return data_train, leafs, data_test


# build LPP Model for each leaf in data_train
# Assume the tree partition indexes of data_train into clusters C_1, ..., C_{2^{ht}} with centers m_1, ..., m_{2^{ht}} is given in leafs
# first project each C_i to local PCA with dimension kd_PCA  
# then continue to construct the local LPP frames A_1, ..., A_{2^{ht}} in G(kd_data, kd_LPP) using supervised affinity
def LPP_BuildDataModel(data_train, leafs, kd_PCA, kd_LPP):
    # Input:
    #   data_train = the training data set
    #   leafs = the tree partition indexes of data_train into clusters C_1, ..., C_{2^{ht}}
    #   kd_PCA = the initial PCA embedding dimension
    #   kd_LPP = the LPP embedding dimension 
    # Output:
    #   Seq = the LPP frames corresponding to each cluster in data_train, labeling the correponding Grassmann equivalence class

    # obtain the dimension of each sample in data_train["x"]
    kd_data = len(data_train["x"][0])
    # initialize the LPP frames A_1,...,A_{2^{ht}}
    Seq = np.zeros((len(leafs), kd_data, kd_LPP))
    # build LPP Model for each leaf
    # input: data, indx, leafs
    for k in range(len(leafs)):
        # form the data_train subsample the k-th cluster
        data_train_x_k = [data_train["x"][_] for _ in leafs[k]]
        data_train_y_k = [data_train["y"][_] for _ in leafs[k]]
        # augment the data_train_x_k and data_train_y_k by GMM sampling and pre-trained learning model prediction
        doAugment = 1
        if doAugment:
            # bulid a Gaussian mixture model on data_train_x_k
            # sample from this GMM enough number of training data points, form data_train_x_k
            # use the pre-trained learning model, predict labels for the newly generated training set
            number_samples_additional = 200 # the number of additional samples
            learning_model = 'cifar10vgg' #'GMM' # the pre-trained learning model
            data_train_x_k_additional, data_train_y_k_additional = GMM_TrainingDataAugmentation(data_train_x_k, 
                                                                                                data_train_y_k, 
                                                                                                number_samples_additional, 
                                                                                                learning_model)
            data_train_x_k.extend(data_train_x_k_additional)
            data_train_y_k.extend(data_train_y_k_additional)
        # do an initial PCA first, for the k-th cluster, so data_train_x_k dimension is reduced to kd_PCA
        pca = PCA()
        pca.fit(data_train_x_k)
        PCA_k = pca.components_
        data_train_x_k = np.matmul(data_train_x_k, np.array([PCA_k[_] for _ in range(kd_PCA)]).T)
        # then do LPP for the PCA embedded data_train_x_k and reduce the dimension to kd_LPP
        # construct the supervise affinity matrix S
        between_class_affinity = 0
        S_k = affinity_supervised(data_train_x_k, data_train_y_k, between_class_affinity)
        # construct the graph Laplacian L and degree matrix D
        L_k, D_k = graph_laplacian(S_k)
        # do LPP
        A_k, LAMBDA = LPP(data_train_x_k, L_k, D_k)
        LPP_k, R = np.linalg.qr(A_k)        
        # obtain the frame Seq(:,:,k)
        Seq[k] = np.matmul(np.array([PCA_k[_] for _ in range(kd_PCA)]).T, np.array([LPP_k[_] for _ in range(1, kd_LPP+1)]).T)
        print("frame ",k+1," size=(", len(Seq[k]),",",len(Seq[k][0]), "), Stiefel = ", np.linalg.norm(np.array(np.matmul(Seq[k].T, Seq[k]))-np.array(np.diag(np.ones(kd_LPP)))))

    return Seq


# Test the LPP piecewise linear embedding model and the interpolated piecewise linear embedding model
# Using k-nearest neighbor classification method
# Also compare with the nearest neighbor classification in the original dimension
def LPP_NearestNeighborTest():
    # select which dataset to work on
    doMNIST = 0
    doCIFAR10 = 1
    # load data
    data_original_train, data_original_test = load_data(doMNIST, doCIFAR10)
    # the data preprocessing projection dimension
    d_pre = 256
    # the PCA embedding dimension = kd_PCA
    kd_PCA = 128
    # the LPP embedding dimension = kd_LPP
    kd_LPP = 100
    # train_size = the training data size
    train_size = 100 * (2**8)
    # ht = the partition tree height
    ht = 8
    # test_size = the test data size
    test_size = 100

    # obtain the train, test sets in nwpu and the LPP frames Seq(:,:,k) for each cluster with indexes in leafs
    data_train, leafs, data_test = LPP_ObtainData(data_original_train, data_original_test, d_pre, kd_LPP, kd_PCA, train_size, test_size, ht)
    Seq = LPP_BuildDataModel(data_train, leafs, kd_PCA, kd_LPP)

    # all these LPP Stiefel frames are on St(n, p)
    n = len(Seq[0])
    p = len(Seq[0][0])
        
    # data original dimension kd_data
    kd_data = len(data_train["x"][0])
        
    # find m_1, ..., m_{2^{ht}}, the means of the chosen clusters
    m = np.zeros((2**ht, kd_data))
    for k in range(2**ht):
        m[k] = np.mean([data_train["x"][_] for _ in leafs[k]], axis=0)

    # set the sequence of interpolation numbers and the threshold ratio for determining the interpolation number
    interpolation_number_seq = np.ones(test_size)
    ratio_threshold = 1.2 # the ratio for determinining the interpolation_number, serve as a tuning parameter
    ratio_seq = np.zeros((test_size, 2)) # the sequence of second smallest (or largest) to-center distance over smallest to-center distance, for tuning ratio_threshold
   
    K = 1e-8 # the scaling coefficient for calculating the weights w = e^{-K distance^2}
    k_nearest_neighbor = 1 # the parameter k for k-nearest-neighbor classification
   
    classified_o = np.zeros(test_size) # list of classified/not classified projections for using the original data point and nearest cluster
    classified_agg_o = np.zeros(test_size) # list of classified/not classified projections for using the original data point and nearest (interpolation_number) clusters
    classified_bm = np.zeros(test_size) # list of classified/not classified projections for using the nearest frame, benchmark
    classified_c = np.zeros(test_size) # list of classified/not classified projections for using the Grassmann center method
   
    doGrassmannpFCenter = 0 # do or do not do projected Frobenius center of mass for Grassmannian frame
    doStiefelEuclidCenter = 1 # do or do not do Euclid center of mass for Stiefel frame 
    doGD = 0 # do or do not do GD for finding projected Frobenius center of mass
   
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
        frames = np.zeros((interpolation_number, kd_data, kd_LPP))
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
        threshold_gradnorm = 1e-4
        threshold_fixedpoint = 1e-4
        threshold_checkonGrassmann = 1e-10
        threshold_checkonStiefel = 1e-10
        threshold_logStiefel = 1e-4
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
        # output the result
        print("original dimension classified =", isclassified_o, ", original dimension aggregate classified =", isclassified_agg_o, ", benchmark classified =", isclassified_bm, ", center mass classfied =", isclassified_c)

    # summarize the final result
    cpu_time_end = time.process_time()
    print("\n******************** CONCLUSION ********************")
    print("\ncpu runtime for testing = ", cpu_time_end - cpu_time_start, " seconds \n")
    print("\noriginal dimension classification rate = ", (sum(classified_o)/test_size)*100, "%")
    print("\noriginal dimension aggregate classification rate =", (sum(classified_agg_o)/test_size)*100, "%")
    print("\nbenchmark correct classification rate = ", (sum(classified_bm)/test_size)*100, "%")
    print("\ncenter mass correct classification rate =", (sum(classified_c)/test_size)*100, "%\n")

    return None


# given a set of training_data_original_x with labels training_data_original_y
# fit from them a GMM model and sample from this GMM model a given number of additional training samples training_data_additional_x 
# with training_data_additional_x, using a pre-trained learning_model, label each additional sample and produce corresponding labels training_data_additional_y
def GMM_TrainingDataAugmentation(training_data_original_x, training_data_original_y, number_samples_additional, learning_model):

    # obtain the number of different labels in training_data_original_y
    num_classes = len(set(training_data_original_y))

    # fit train_data_original_x using a GMM model
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=num_classes).fit(training_data_original_x)

    # using GMM, generate an additional set of training_data_additional_x and predict training_data_additional_y
    training_data_additional_x_, y = gmm.sample(number_samples_additional)
    if learning_model == 'cifar10vgg':
        model = cifar10vgg()
        training_data_additional_x__ = np.reshape(training_data_additional_x_.flatten(), (200, 32, 32, 3))
        predicted_x = model.predict(training_data_additional_x__)
        training_data_additional_y = np.argmax(predicted_x, 1)
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
    
    # do test correctness of the specific functions developed
    doruntest=0
    if doruntest:
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
    
 
    
    # do the LPP analysis on different datasets
    dorunfile = 1
    
    if dorunfile:
        LPP_NearestNeighborTest()


