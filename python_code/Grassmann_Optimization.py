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
        # find the value and grad of the projected Frobenius distance center of mass function f(A)=\sum_{k=1}^m w_k |AA^T-A_kA_k^T|_F^2 on G_{n,p}
        A = Y
        m = len(self.omega)
        n = len(A)
        p = len(A[0])
        value = 0
        for k in range(m):
            Mtx = np.matmul(A, A.T) - np.matmul(self.Seq[k], self.Seq[k].T)
            value = value + self.omega[k]*((np.linalg.norm(Mtx))**2)
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
        n = len(self.Seq)
        p = len(self.Seq[0])
        total_weight = sum(self.omega)
        Mtx = np.zeros((n, n), dtype=float)
        for k in range(m):
            Mtx = Mtx + np.matmul(self.Seq[k], self.Seq[k].T)*(self.omega[k]/total_weight)
        Q, D, Q1 = np.linalg.svd(Mtx, full_matrices=True)
        eye_p = np.zeros((p, p), dtype=float)
        np.fill_diagonal(eye_p, 1)
        I = np.pad(eye_p, ((0,n-p), (0,0)), 'constant', constant_values = (0,0))
        pF_Center = np.matmul(Q , I)
        value, grad = self.Center_Mass_function_gradient_pFrobenius(pF_Center)
        return pF_Center, value, grad



    
"""
################################ MAIN TESTING FILE #####################################
################################ FOR DEBUGGING ONLY #####################################

testing the Optimization Calculus on Grassmann manifolds
"""

if __name__ == "__main__":
    
    
    