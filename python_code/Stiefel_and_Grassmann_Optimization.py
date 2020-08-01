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
    
    
    # calculate the function value and the gradient on Stiefel manifold St(p, n) 
    # of the Euclidean center of mass function f_F(A)=\sum_{k=1}^m w_k \|A-A_k\|_F^2
    def Center_Mass_function_gradient_Euclid(self, Y):
        # identify m
        m = len(self.omega)
        n = len(self.Seq[0])
        p = len(self.Seq[0][0])
        # evaluate f
        f = 0
        for i in range(m):
            f = f + self.omega[i]*((np.linalg.norm(Y-self.Seq[i]))**2)
        # evaluate gradf
        gradf = np.zeros((n, p), dtype=float)
        for i in range(m):
            gradf = gradf + 2*self.omega[i]*((Y-self.Seq[i])-np.matmul(np.matmul(Y, (Y-self.Seq[i]).T), Y));
        return f, gradf
    
    # directly calculate the Euclidean center of mass that is the St(p, n) minimizer of f_F(A)=\sum_{k=1}^m w_k\|A-A_k\|_F^2, 
    # according to our elegant lemma based on SVD
    def Center_Mass_Euclid(self):
        # identify m, n and p
        m = len(self.omega)
        n = len(self.Seq[0])
        p = len(self.Seq[0][0])
        # form B = \sum_{k=1}^m w_k A_k
        B = np.zeros((n, p), dtype=float)
        for i in range(m):
            B = B + self.omega[i] * self.Seq[i]
        # do svd on B
        O1, D, O2 = np.linalg.svd(B, full_matrices=True)
        D = np.diag(D)
        # form the Euclid Center of Mass according to our elegant lemma based on SVD
        Mtx = np.zeros((p, n), dtype=float)
        for j in range(p):
            Mtx[j][j]=1
        Mtx = Mtx.T
        Euclid_Center = np.matmul(np.matmul(O1, Mtx), O2)
        # evaluate f_F(A)=\sum_{k=1}^m w_k\|A-A_k\|_F^2 at the center and its grad norm
        value, grad = self.Center_Mass_function_gradient_Euclid(Euclid_Center)
        gradnorm = np.linalg.norm(grad)
        return Euclid_Center, value, gradnorm


    # test if the given matrix Y is on the Stiefel manifold St(p, n)
    def CheckOnStiefel(self, Y):
        # Y is the matrix to be tested, threshold is a threshold value, if \|Y^TY-I_p\|_F < threshold then return true
        # first turn Y into an array   
        Y = np.array(Y)
        n = len(Y)
        p = len(Y[0])
        # form I_p matrix
        eye_p = np.zeros((p, p), dtype=float)
        np.fill_diagonal(eye_p, 1)
        # compute Y^T*Y-I_p
        Mtx = np.matmul(Y.T, Y) - eye_p
        # compute \|Y^T*Y-I_p\|_F
        distance = np.linalg.norm(Mtx)
        if distance <= self.threshold_checkonStiefel:
            ifStiefel = True
        else:
            ifStiefel = False
        return ifStiefel, distance


    # test if the given matrix H is on the tangent space of Stiefel manifold T_Y St(p, n)
    def CheckTangentStiefel(self, Y, H):
        # H is the matrix to be tested, threshold is a threshold value, if \|Y^TH+H^TY\| < threshold then return true
        # first turn Y and H into an array   
        Y = np.array(Y)
        H = np.array(H)
        n = len(Y)
        p = len(Y[0])
        n_H = len(H)
        p_H = len(H[0])
        # check if H is tangent to St(p, n) at Y
        if n==n_H and p==p_H:
            Mtx = np.add(np.matmul(Y.T, H), np.matmul(H.T, Y))
            distance = np.linalg.norm(Mtx)
            if distance <= self.threshold_checkonStiefel:
                ifTangentStiefel = True
            else:
                ifTangentStiefel = False
        else:
            ifTangentStiefel = False
        return ifTangentStiefel, distance


    # calculate the projection onto tangent space of Stiefel manifold St(p, n)
    def projection_tangent(self, Y, Z):
        # Pi_{T, Y}(Z) projects matrix Z of size n by p onto the tangent space of St(p, n) at point Y\in St(p, n)
        # returns the tangent vector prj_tg on T_Y(St(p, n))
        # first turn Y into an array   
        Y = np.array(Y)
        n = len(Y)
        p = len(Y[0])
        # compute (Y' * Z - Z' * Y)/2        
        skew = np.subtract(np.matmul(Y.T, Z), np.matmul(Z.T, Y))/2
        # form I_n matrix
        eye_n = np.zeros((n, n), dtype=float)
        np.fill_diagonal(eye_n, 1)        
        # compute the projection
        prj_tg = np.add(np.matmul(Y, skew), np.matmul(np.subtract(eye_n, np.matmul(Y, Y.T)), Z))
        return prj_tg



"""
################################ MAIN TESTING FILE #####################################
################################ FOR DEBUGGING ONLY #####################################

testing the Optimization Calculus on Stiefel and Grassmann manifolds
"""

if __name__ == "__main__":
    
    # number of frames to find center-of-mass
    m = 3
    # set the sequence of Stiefel matrices in St(p, n)
    n = 4
    p = 2
    Seq = np.array([[[0 for _ in range(p)] for _ in range(n)] for _ in range(m)])
    Seq[0] = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])
    Seq[1] = np.array([[1, 0], [0, 0], [0, 0], [0, 1]])
    Seq[2] = np.array([[0, 0], [0, 1], [0, 0], [1, 0]])
    # set the weights
    omega = np.array([1, 3, 4])
    omega = omega.astype(np.float)
    # set the threshold numbers
    threshold_gradnorm = 1e-4
    threshold_fixedpoint = 1e-4
    threshold_checkonStiefel = 1e-10
    threshold_checkonGrassmann = 1e-10
    threshold_logStiefel = 1e-4
    # set the Stiefel Optimization object
    StiefelOpt = Stiefel_Optimization(omega, Seq, threshold_gradnorm, threshold_fixedpoint, threshold_checkonStiefel, threshold_logStiefel)
    
    # do check Stiefel and check tangent Stiefel
    docheckStiefel = 0
    if docheckStiefel:
        Y = Seq[1]
        H = Seq[0]
        ifStiefel, distance = StiefelOpt.CheckOnStiefel(Y)
        ifTangentStiefel, distance_tg = StiefelOpt.CheckTangentStiefel(Y, H)
        print("\nY= ", Y, "\nH=", H)
        print("\nifStiefel Y=", ifStiefel, "\nifTangentStiefel H to Y=", ifTangentStiefel)
        H_prj = StiefelOpt.projection_tangent(Y, H)
        ifTangentStiefel, distance_tg = StiefelOpt.CheckTangentStiefel(Y, H_prj)
        print("\nH_prj=", H_prj)
        print("\nifTangentStiefel H_prj to Y=", ifTangentStiefel, "\n")

        
    # do complete special orthogonal
    doComplete_SpecialOrthogonal = 0
    if doComplete_SpecialOrthogonal:
        A = Seq[2]
        Q = StiefelOpt.Complete_SpecialOrthogonal(A)
        print("A=\n", A, "\nA'*A=\n", np.matmul(A.T, A), "\n")
        print("Q=\n", Q, "\nQ'*Q=\n", np.matmul(Q.T, Q), "\nQ*Q'=\n", np.matmul(Q, Q.T), "\n")
        ifStiefel_A, distance_A = StiefelOpt.CheckOnStiefel(A)
        ifStiefel_Q, distance_Q = StiefelOpt.CheckOnStiefel(Q)
        print("Check A on Stiefel = ", ifStiefel_A, ", distance = ", distance_A, "\n")
        print("Check Q on Stiefel = ", ifStiefel_Q, ", distance = ", distance_Q, "\n")
        
        
    # do Euclid center of mass
    doCenterMassEuclid = 1
    if doCenterMassEuclid:
        center, value, gradnorm = StiefelOpt.Center_Mass_Euclid()
        for i in range(m):
            print("frame ", i+1, "weight is ", omega[i], " matrix is \n", Seq[i], "\n")
        print("center is \n", center, "\n")
        print("function f_F(A)=\sum_{k=1}^m w_k\|A-A_k\|_F^2\nvalue is ", value, " gradnorm is ", gradnorm, "\n")