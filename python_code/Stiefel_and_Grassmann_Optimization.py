#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 15:44:40 2020

@author: Wenqing Hu (Missouri S&T)

Title: Stiefel and Grassmann Optimization

"""

import numpy as np


"""
Optimization Calculus on Stiefel Manifold 

contains various functions for operating optimization calculus and related geometries on Stiefel Manifold St(p, n)
"""
class Stiefel_Optimization:
    
    def __init__(self,
                 omega,                 # the weight sequence
                 Seq,                   # the sequence of pointes on St(p, n)
                 threshold_gradnorm,    # the threshold for gradient norm when using GD
                 threshold_fixedpoint,  # the threshold for fixed-point iteration for average
                 threshold_checkonStiefel,  # the threshold for checking if iteration is still on St(p, n)
                 threshold_logStiefel   # the threshold for calculating the Stiefel logarithmic map via iterative method)
                 ):
        self.omega=omega
        self.Seq=Seq
        self.thereshold_gradnorm=threshold_gradnorm
        self.threshold_checkonStiefel=threshold_checkonStiefel
        self.threshold_logStiefel=threshold_logStiefel
        
    # given the matrix A in St(p, n), complete it into Q = [A B] in SO(n)
    def Complete_SpecialOrthogonal(self, A):
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





"""
################################ MAIN TESTING FILE #####################################
################################ FOR DEBUGGING ONLY #####################################

testing the Optimization Calculus on Stiefel and Grassmann manifolds
"""

if __name__ == "__main__":
    
    # number of frames to find center-of-mass
    number = 3
    # set the sequence of Stiefel matrices in St(p, n)
    n = 4
    p = 2
    Seq = np.array([[[0 for _ in range(p)] for _ in range(n)] for _ in range(number)])
    Seq[0] = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])
    Seq[1] = np.array([[1, 0], [0, 0], [0, 0], [0, 1]])
    Seq[2] = np.array([[0, 0], [0, 1], [0, 0], [1, 0]])
    # set the weights
    omega = [1, 1, 1]
    # set the threshold numbers
    threshold_gradnorm = 1e-4
    threshold_fixedpoint = 1e-4
    threshold_checkonStiefel = 1e-10
    threshold_checkonGrassmann = 1e-10
    threshold_logStiefel = 1e-4
    # set the Stiefel Optimization object
    StiefelOpt = Stiefel_Optimization(omega, Seq, threshold_gradnorm, threshold_fixedpoint, threshold_checkonStiefel, threshold_logStiefel)
    
    # do complete special orthogonal
    doComplete_SpecialOrthogonal = 1
    if doComplete_SpecialOrthogonal:
        A = Seq[2]
        Q = StiefelOpt.Complete_SpecialOrthogonal(A)
        print("A=\n", A, "\nA'*A=\n", np.matmul(A.T, A))
        print("Q=\n", Q, "\nQ'*Q=\n", np.matmul(Q.T, Q), "\nQ*Q'=\n", np.matmul(Q, Q.T))