#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 20:55:11 2020

@author: Wenqing Hu (Missouri S&T)
Title: Grassmann Optimization
"""

import numpy as np


"""
Optimization Calculus on Grassmann Manifold 

contains various functions for operating optimization calculus and related geometries on Grassmann Manifold Gr(p, n)
"""
class Grassmann_Optimization:
    
    def __init__(self,
                 omega,                 # the weight sequence
                 Seq,                   # the sequence of pointes on Gr(p, n)
                 threshold_gradnorm,    # the threshold for gradient norm when using GD
                 threshold_fixedpoint,  # the threshold for fixed-point iteration for average
                 threshold_checkonGrassmann,  # the threshold for checking if iteration is still on Gr(p, n)
                 ):
        self.omega = omega
        self.Seq = Seq
        self.thereshold_gradnorm = threshold_gradnorm
        self.threshold_fixedpoint = threshold_fixedpoint
        self.threshold_checkonGrassmann = threshold_checkonGrassmann
        
    
    def Center_Mass_function_gradient_pFrobenius(self, Y):
        # find the value and grad of the projected Frobenius distance center of mass function f(A) = \sum_{k=1}^m w_k |AA^T-A_kA_k^T|_F^2 on Gr(p, n)
        A = Y
        m = len(self.omega)
        n = len(A)
        p = len(A[0])
        # calculate the value f(A) = \sum_{k=1}^m w_k |AA^T-A_kA_k^T|_F^2 on Gr(p, n)
        value = 0
        for k in range(m):
            Mtx = np.matmul(A, A.T) - np.matmul(self.Seq[k], self.Seq[k].T)
            value = value + self.omega[k]*((np.linalg.norm(Mtx))**2)
        # calculate grad f(A) =  \sum_{k=1}^m (I-AA^T)(2 w_k A-4 w_k A_kA_k^TA)
        grad = np.zeros((n, p), dtype=float)
        for k in range(m):
            M1 = A * (2 * self.omega[k])
            M2 = np.matmul(np.matmul(self.Seq[k], self.Seq[k].T), A) * (4 * self.omega[k])
            grad = grad + M1 - M2;
        grad = grad - np.matmul(np.matmul(A, A.T), grad)
        return value, grad
    
    def Center_Mass_pFrobenius(self):
        # directly calculate the center of mass on Gr(p,n) with respect to projected Frobenius norm
        m  = len(self.omega)
        n = len(self.Seq[0])
        p = len(self.Seq[0][0])
        # total weight is the sum of all weights \sum_{k=1}^m w_k
        total_weight = sum(self.omega)
        # compute the matrix Mtx = \sum_{k=1}^m (w_k/total weight)A_kA_k^T
        Mtx = np.zeros((n, n), dtype=float)
        for k in range(m):
            Mtx = Mtx + np.matmul(self.Seq[k], self.Seq[k].T)*(self.omega[k]/total_weight)
        # do an svd decomposition Q D Q1 = Mtx
        Q, D, Q1 = np.linalg.svd(Mtx, full_matrices=True)
        # obtain the identity matrix eye_p = diag(1, 1, ..., 1) in dimension p
        eye_p = np.zeros((p, p), dtype=float)
        np.fill_diagonal(eye_p, 1)
        # form I = (eye_p; 0_{n-p, p})
        I = np.pad(eye_p, ((0,n-p), (0,0)), 'constant', constant_values = (0,0))
        # the p-Frobenius center of mass is given by QI
        pF_Center = np.matmul(Q , I)
        # evaluate the value and gradient on Gr(p, n) of the p-Frobenius center of mass
        value, grad = self.Center_Mass_function_gradient_pFrobenius(pF_Center)
        
        return pF_Center, value, grad



    
"""
################################ MAIN TESTING FILE #####################################
################################ FOR DEBUGGING ONLY #####################################

testing the Optimization Calculus on Grassmann manifolds
"""

if __name__ == "__main__":
    # number of frames to find center-of-mass
    m = 3
    # set the sequence of Stiefel matrices in St(p, n), treated as elements on Gr(p, n)
    n = 5
    p = 4
    Seq = np.array([[[0 for _ in range(p)] for _ in range(n)] for _ in range(m)])
    Seq[0] = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    Seq[1] = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    Seq[2] = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # set the weights
    omega = np.array([1, 10, 100])
    omega = omega.astype(np.float)
    # set the threshold numbers
    threshold_gradnorm = 1e-7
    threshold_fixedpoint = 1e-4
    threshold_checkonGrassmann = 1e-10
    # set the Grassmann Optimization object
    GrassmannOpt = Grassmann_Optimization(omega, Seq, threshold_gradnorm, threshold_fixedpoint, threshold_checkonGrassmann)
    
    # do p-Frobenious center of mass
    doCenterMasspFrobenius = 1
    if doCenterMasspFrobenius:
        center, value, grad = GrassmannOpt.Center_Mass_pFrobenius()
        for i in range(m):
            print("frame ", i+1, "weight is ", omega[i], " matrix is \n", Seq[i], "\n")
        print("center is \n", center, "\n")
        print("function f_F(A)=\sum_{k=1}^m w_k\|A-A_k\|_F^2\nvalue is ", value, "\ngradnorm is ", np.linalg.norm(grad), "\n")    
    
    
    
    