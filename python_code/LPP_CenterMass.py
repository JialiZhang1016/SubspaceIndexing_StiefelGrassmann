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



# k-nearest neighbor classfication
# given test data x and label y, find in a training set (X, Y) the k-nearest points x1,...,xk to x, and classify x as majority vote on y1,...,yk
# if the classification is correct, return 1, otherwise return 0
def knn(x_test, y_test, X_train, Y_train, k):
    m = len(Y_train)
    if k>m:
        k=m
    # find the first k-nearest neighbor
    dist = [np.linalg.norm(np.array(x_test)-np.array(X_train[i])) for i in range(m)]
    print(dist)
    indexes, dist_sort = zip(*sorted(enumerate(dist), key=itemgetter(1))) 
    print(indexes, dist_sort)
    # do a majority vote on the first k-nearest neighbor
    label = [Y_train[indexes[_]] for _ in range(k)]
    vote = pd.value_counts(label)
    print(vote)
    # class_predict is the predicted label based on majority vote
    class_predict = vote.index[0]
    if class_predict == y_test:
        isclassified = 1
    else:
        isclassified = 0
    return isclassified






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
    
    x_test = [1, 2]
    y_test = 1
    X_train = [[2, 3], [1, 2], [6, 5]]
    Y_train = [3, 1, 3]
    k = 1
    isclassified = knn(x_test, y_test, X_train, Y_train, k)
    print("isclassified=", isclassified)