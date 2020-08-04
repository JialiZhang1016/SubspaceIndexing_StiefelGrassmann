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
    mtx_L = np.matmul(np.matmul(X, L), X.T)
    print("mtx_L =", mtx_L)
    # calculate mtx_D = X' * D * X
    mtx_D = np.matmul(np.matmul(X, D), X.T)
    print("mtx_D =", mtx_D)
    # solve the generalized eigenvalue problem mtx_L W = LAMBDA mtx_D W
    LAMBDA, W = eigh(mtx_L, mtx_D, eigvals_only=False)
    # sort the eigenvalues in a descending order
    SORT_ORDER, LAMBDA = zip(*sorted(enumerate(LAMBDA), key=itemgetter(1), reverse=True)) 
    # reorder the generalized eigenvector matrix W according to SORT_ORDER
    W = [W[SORT_ORDER[_]] for _ in range(len(D))]
    return W, LAMBDA 
    
 
# construct the graph laplacian L and the degress matrix D from the given affinity matrix S 
def graph_laplacian(S):
    # first turn S into an array
    S = np.array(S)
    # compute the D matrix
    D = np.diag(sum(S, 0))
    L = D - S
    return L, D





"""
################################ MAIN RUNNING FILE #####################################

LPP analysis based on Grassmann center of mass calculation
"""

if __name__ == "__main__":
    
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