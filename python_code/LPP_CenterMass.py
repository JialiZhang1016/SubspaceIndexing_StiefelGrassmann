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


# load the data set, nwpu-aerial-images, MNIST or CIFAR-10
def load_data(doNWPU, doMNIST, doCIFAR10):
    
    if doNWPU:
        # load the nwpu-aerial-images dataset
        # structure: 
        #   x: [31500×4096 double]
        #   y: [31500×1 double]
        data = {"x": [0], "y": [1]}

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
        data = {"x": [], "y": []}
        # first turn the matrices of x_train_pre and x_test_pre to 28 x 28 = 784 dimensional vectors
        for i in range(60000):
            data["x"].append(np.reshape(x_train[i], 784))
            data["y"].append(y_train[i])
        for i in range(10000):
            data["x"].append(np.reshape(x_test[i], 784))
            data["y"].append(y_test[i])

    if doCIFAR10:
        # load the CIFAR-10 dataset, data from https://www.cs.toronto.edu/~kriz/cifar.html
        # structure: 
            # each cifar10_k, k=1,...,5
            #          data: [10000×3072 uint8]
            #        labels: [10000×1 uint8]
            #   batch_label: 'training batch k of 5'
            # cifar10_test
            #          data: [10000×3072 uint8]
            #        labels: [10000×1 uint8]
            #   batch_label: 'testing batch 1 of 1'
        data = {"x": [0], "y": [1]}
        
    return data


# k-nearest neighbor classfication
# given test data x and label y, find in a training set (X, Y) the k-nearest points x1,...,xk to x, and classify x as majority vote on y1,...,yk
# if the classification is correct, return 1, otherwise return 0
def knn(x_test, y_test, X_train, Y_train, k):
    m = len(Y_train)
    if k>m:
        k=m
    # find the first k-nearest neighbor
    dist = [np.linalg.norm(np.array(x_test)-np.array(X_train[i])) for i in range(m)]
    #print(dist)
    indexes, dist_sort = zip(*sorted(enumerate(dist), key=itemgetter(1))) 
    #print(indexes, dist_sort)
    # do a majority vote on the first k-nearest neighbor
    label = [Y_train[indexes[_]] for _ in range(k)]
    vote = pd.value_counts(label)
    #print(vote)
    # class_predict is the predicted label based on majority vote
    class_predict = vote.index[0]
    if class_predict == y_test:
        isclassified = 1
    else:
        isclassified = 0
    return isclassified


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
    SORT_ORDER, LAMBDA = zip(*sorted(enumerate(LAMBDA), key=itemgetter(1), reverse=True)) 
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
    #print("S1=", S1)
    # utilize supervised info
    # first turn Y into a 2-d array
    Y = [[Y[_]] for _ in range(len(Y))]
    id_dist = cdist(Y, Y, 'euclidean')
    #print("id_dist=", id_dist)
    S2 = S1 
    for i in range(len(X)):
        for j in range(len(X)):
            if id_dist[i][j] != 0:
                S2[i][j] = between_class_affinity
    # obtain the supervised affinity S
    S = S2
    return S


# given the matrix A in St(p, n), complete it into Q = [A B] in SO(n)
def Complete_SpecialOrthogonal(A):
    # first turn the matrix A into an array
    A = np.array(A)
    # the number of rows in A
    n = len(A)
    # the number of columns in A
    p = len(A[0])
    # full svd decomposition of A
    O1, D, O2 = np.linalg.svd(A, full_matrices=True)
    D = np.diag(D)
    # extend O2 to O2_ext = [O2 zeros(p, n-p); zeros(n-p, p) eye(n-p)]
    O2_ext = np.pad(O2, ((0,n-p),(0,n-p)), 'constant', constant_values = (0,0))
    for j in range(n-p):
        O2_ext[p+j][p+j]=1
    # compute Q = O1 * O2_ext, if det(Q)=-1, make it +1 
    Q = np.matmul(O1, O2_ext)
    if np.linalg.det(Q)<0:
        Q[0:n-1, p] = -Q[0:n-1, p]
    return Q


# Sample a training dataset data_train from the data set, data_train = (data_train.x, data_train.y)
# Set the partition tree depth = ht
# Tree partition nwpu_train into clusters C_1, ..., C_{2^{ht}} with centers m_1, ..., m_{2^{ht}}
# first project each C_i to local PCA with dimension kd_PCA  
# then continue to construct the local LPP frames A_1, ..., A_{2^{ht}} in G(kd_data, kd_LPP) using supervised affinity
# Sample a test dataset data_test from the data set for testing purposes, data_test = (data_test.x, data_test.y)
def LPP_train(data, d_pre, kd_LPP, kd_PCA, train_size, ht, test_size):
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
    #   Seq = the LPP frames corresponding to each cluster in data_train, labeling the correponding Grassmann equivalence class
    
    # read the data into inputs and labels
    data_x = data["x"]
    data_y = data["y"]
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    
    # do an initial PCA on data
    pca = PCA()
    pca.fit(data_x)
    A0 = pca.components_
    # bulid a given dimensional d_pre embedding of data_x into new data_x, for faster computation only
    data_x = np.matmul(data_x, np.array([A0[_] for _ in range(d_pre)]).T)
    
    # n_data is the number of samples in data_x dataset, kd_data is the original dimension of each sample
    n_data = len(data_x)
    kd_data = len(data_x[0])
    
    indexes = np.random.permutation(n_data) 
    # randomly pick the training sample of size train_size from data.x dataset
    train_indexes = [indexes[_] for _ in range(train_size)]
    # form the data_train dataset
    data_train_x = [data_x[_] for _ in train_indexes]
    data_train_y = [data_y[_] for _ in train_indexes]
    data_train = {"x": data_train_x, "y": data_train_y}
    
    # randomly pick the test sample of size test_size from data dataset, must be disjoint from data_train
    test_indexes = [indexes[_]  for _ in range(train_size, train_size + test_size)]
    # form the data_test dataset
    data_test_x = [data_x[_] for _ in test_indexes]
    data_test_y = [data_y[_] for _ in test_indexes]
    data_test = {"x": data_test_x, "y": data_test_y}
    
    # do an initial PCA on data_train
    pca = PCA()
    pca.fit(data_train_x)
    A0 = pca.components_
    # bulid a kd_PCA dimensional embedding of data_train in x0
    x0 = np.matmul(data_train_x, np.array([A0[_] for _ in range(kd_PCA)]).T)
    # from x0, partition into 2^ht leaf nodes, each leaf node can give samples for a local LPP
    indx, leafs, mbrs = buildVisualWordList(x0, ht)

    # initialize the LPP frames A_1,...,A_{2^{ht}}
    Seq = np.zeros((len(leafs), kd_data, kd_LPP))
    # build LPP Model for each leaf
    doBuildDataModel = 1
    # input: data, indx, leafs
    if doBuildDataModel:
        for k in range(len(leafs)):
            # form the data_train subsample for the k-th cluster
            data_train_x_k = [data_train_x[_] for _ in leafs[k]]
            data_train_y_k = [data_train_y[_] for _ in leafs[k]]
            # do an initial PCA first, for the k-th cluster, so data_train_x_k dimension is reduced to kd_PCA
            pca.fit(data_train_x_k)
            PCA_k = pca.components_
            PCA_k = Complete_SpecialOrthogonal(PCA_k.T).T
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
            Seq[k] = np.matmul(np.array([PCA_k[_] for _ in range(kd_PCA)]).T, np.array([LPP_k[_] for _ in range(kd_LPP)]).T)
            print("frame ",k+1," size=(", len(Seq[k]),",",len(Seq[k][0]), "), Stiefel = ", np.linalg.norm(np.array(np.matmul(Seq[k].T, Seq[k]))-np.array(np.diag(np.ones(kd_LPP)))))

    return data_train, Seq, leafs, data_test



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
    # select which dataset to work on
    doNWPU = 0
    doMNIST = 1 
    doCIFAR10 = 0
    
    if dorunfile:
        # load data
        data = load_data(doNWPU, doMNIST, doCIFAR10)
        # the data preprocessing projection dimension
        d_pre = 256
        # the PCA embedding dimension = kd_PCA
        kd_PCA = 64
        # the LPP embedding dimension = kd_LPP
        kd_LPP = 16
        # train_size = the training data size
        train_size = 100*(2**9)
        # ht = the partition tree height
        ht = 9
        # test_size = the test data size
        test_size = 100

        # obtain the train, test sets in nwpu and the LPP frames Seq(:,:,k) for each cluster with indexes in leafs
        data_train, Seq, leafs, data_test = LPP_train(data, d_pre, kd_LPP, kd_PCA, train_size, ht, test_size)

        print(len(data_train["x"]), len(data_train["x"][0]))
        print(len(data_test["x"]), len(data_test["x"][0]))
        print(len(Seq[0]), len(Seq[0][0]))
    
#   % all these LPP Stiefel frames are on St(n, p)
#   n = size(Seq, 1);
#   p = size(Seq, 2);
#
#   % data original dimension kd_data
#   kd_data = size(data_train.x, 2);
#   
#   % find m_1, ..., m_{2^{ht}}, the means of the chosen clusters
#   m = zeros(kd_data, 2^ht);
#   for k=1:2^ht 
#       m(:, k) = mean(data_train.x(leafs{k}, :), 1);
#   end
#
#   % set the sequence of interpolation numbers and the threshold ratio for determining the interpolation number
#   interpolation_number_seq = ones(test_size, 1);
#   ratio_threshold = 1.001;
#   
#   K = 1e-8; % the scaling coefficient for calculating the weights w = e^{-K distance^2}
#   k_nearest_neighbor = 80; % the parameter k for k-nearest-neighbor classification
#   
#   classified_bm = zeros(test_size, 1); % list of classified/not classified projections for using the nearest frame, benchmark
#   classified_c = zeros(test_size, 1);  % list of classified/not classified projections for using the Grassmann center method
#   
#   doGrassmannpFCenter = 1; % do or do not do projected Frobenius center of mass for Grassmannian frame
#   doStiefelEuclidCenter = 0; % do or do not do Euclid center of mass for Stiefel frame 
#   doGD = 0; % do or do not do GD for finding projected Frobenius center of mass
#   
#   tic;
#   for test_index=1:test_size
#        fprintf("\ntest point %d -----------------------------------------------------------\n", test_index);
#        x = data_test.x(test_index, :);
#        y = data_test.y(test_index);
#        % sort the cluster centers m_1, ..., m_{2^{ht}} by ascending distances to x 
#        dist = zeros(2^ht, 1);
#        for k=1:2^ht
#            dist(k) = norm(x-m(:, k));
#        end
#        [dist_sort, indexes] = sort(dist, 1, 'ascend');
#        % count the number of St(p, n) interpolation clusters for current test point x
#        % interpolation_number = number of frames used for interpolation between cluster LDA frames
#        interpolation_number = 1;
#        for k=2:2^ht
#            if dist_sort(k) <= ratio_threshold * dist_sort(1)
#                interpolation_number = interpolation_number + 1;
#            else
#                break;
#            end    
#        end
#        fprintf("interpolation number = %d\n", interpolation_number);
#        % record the sequence of all interpolation numbers for each test point x
#        interpolation_number_seq(test_index) = interpolation_number;
#        % find the LPP Stiefel projection frames A_k1, ..., A_k{interpolation_number} for the first (interpolation_number) closest clusters to x
#        frames = zeros(kd_data, kd_LPP, interpolation_number);
#        for i=1:interpolation_number
#            frames(:, :, i) = Seq(:, :, indexes(i));
#            end
#            % find the weights w_1, ..., w_{interpolation_number} for the first (interpolation_number) closest clusters to x
#            w = zeros(interpolation_number, 1);
#        for i=1:interpolation_number
#            w(i) = exp(- K * (dist_sort(i))^2);
#        end
#        % collect all indexes in clusters corresponding to the first (interpolation_number) closest clusters to x
#        aggregate_cluster = [];
#        for i=1:interpolation_number
#            aggregate_cluster = union(aggregate_cluster, leafs{indexes(i)});
#        end
#        % project x to A1 x and classify it using k-nearest-neighbor on the projection via A1 of the closest cluster
#        x_test = x * frames(:,:,1);
#        y_test = y;
#        X_train = data_train.x(leafs{indexes(1)}, :) * frames(:,:,1);
#        Y_train = data_train.y(leafs{indexes(1)});
#        isclassified_bm = knn(x_test, y_test, X_train, Y_train, k_nearest_neighbor);
#        classified_bm(test_index) = isclassified_bm;
#        % calculate the center of mass for the (interpolation_number) nearest cluster LPP frames with respect to weights w 
#        threshold_gradnorm = 1e-4;
#        threshold_fixedpoint = 1e-4;
#        threshold_checkonGrassmann = 1e-10;
#        threshold_checkonStiefel = 1e-10;
#        threshold_logStiefel = 1e-4;
#        if doGrassmannpFCenter
#            % do Grassmann center of mass method
#            GrassmannOpt = Grassmann_Optimization(w, frames, threshold_gradnorm, threshold_fixedpoint, threshold_checkonGrassmann);
#            if doGD
#                break;
#            else
#                [center, value, grad] = GrassmannOpt.Center_Mass_pFrobenius;
#            end
#        else
#            % do Stiefel center of mass method
#            StiefelOpt = Stiefel_Optimization(w, frames, threshold_gradnorm, threshold_fixedpoint, threshold_checkonStiefel, threshold_logStiefel);
#            if doStiefelEuclidCenter
#                if doGD
#                    break;
#                else
#                    [center, value, gradnorm] = StiefelOpt.Center_Mass_Euclid;
#                end
#            else
#                break;
#            end
#        end
#        % project x to center x and classify it using k-nearest-neighbor on the projection via center of all (interpolation number) clusters
#        x_test = x * center;
#        y_test = y;
#        X_train = data_train.x(aggregate_cluster, :) * center;
#        Y_train = data_train.y(aggregate_cluster);    
#        isclassified_c = knn(x_test, y_test, X_train, Y_train, k_nearest_neighbor);
#        classified_c(test_index) = isclassified_c;
#        % output the result
#        fprintf("benchmark classified = %d, center mass classfied = %d\n", isclassified_bm, isclassified_c);
#        end
#    toc;
#
#    fprintf("benchmark correct classification rate = %f %%, center mass correct classification rate = %f %%\n", (sum(classified_bm)/test_size)*100, (sum(classified_c)/test_size)*100);
