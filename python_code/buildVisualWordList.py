#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 17:43:37 2020
@author: Wenqing Hu (Missouri S&T)
Title: Parition the given dataset into kd-tree with prescribed height ht
Note: This is the python version of the matlab code BuildVisualWordList.m originally by Prof. Zhu Li at UMKC 
"""

import numpy as np
from operator import itemgetter
import math

############################################################################
# function buildVisualWordList()
# visual code word and inverted list approach with kd-tree in fast dub detection
# input:
#   x - n x d data points
#   ht - kd-tree height
# output:
#   indx - indx structure with dim and val of cuts
#   leafs - leaf nodes of offs of x. 
#   mbrs - min bounding rectangles
############################################################################

def buildVisualWordList(x, ht):
    
    # turn x into an array of dimension n times d, x is an array
    x = np.array(x)
    print("x=", x)
    print("ht=", ht)
    # var
    n = len(x)
    print("n=", n)
    kd = len(x[0])
    nNode = 2**(ht+1) - 1 
    nLeafNode = 2**ht
        
    # intermediate storages, offs is a list of lists
    offs = []
    for i in range(nNode):
        offs.append([])
    print("offs=", offs)
    
    # initialize indx as a dictionary {"d_cuts": [store cut dimensions], "v_cuts": [store cut values]}
    indx = {"d_cuts": [], "v_cuts": []}
    
    # first cut
    # cut at dimension 0
    indx['d_cuts'].append(0) 
    # sort the x-value at first dimension, sv, soffs are lists
    soffs, sv = zip(*sorted(enumerate(x.T[0]), key=itemgetter(1))) 
    sv = list(sv)
    soffs = list(soffs)
    print("sv=", sv)
    print("soffs=", soffs)
    indx['v_cuts'].append(sv[math.floor(n/2)-1])
    print("indx=", indx)
    
    # split at mv, left child x(:,1) <= mv, right child x(:,1) > mv
    offs[1]=soffs[1-1 : math.floor(n/2)-1] 
    offs[2]=soffs[math.floor(n/2)+1-1 : n-1]
    print("offs=", offs)
    
    for h in range(ht-1):
        print("h=", h)
        # parents nodes at height h+1
        for k in range(2**(h+1), 2**(h+1+1)):
            print("k=", k)
            # compute covariance
            offs_k = offs[k-1]
            print("offs_k=", offs_k)
            nk = len(offs_k)
            # median offs 
            moffs = math.floor(nk/2-1) 
            # pick all rows of x with indexes from offs_k
            x_offs_k = []
            for i in offs_k:
                x_offs_k.append(x[i])
            print("x_offs_k=", x_offs_k)
            # compute the covariance of x along the dimensions in offs_k, find the dimension of the maximal variance
            sk = np.var(x_offs_k, 1);
            print("sk=", sk)
            max_s = max(sk)
            sk = list(sk)
            d_cut_index = sk.index(max(sk))
            d_cut = offs_k[d_cut_index]
            print("d_cut_index=", d_cut_index, ", d_cut=", d_cut)
    
    return indx, offs #, leafs, mbrs





"""
################################ MAIN TESTING FILE #####################################
################################ FOR DEBUGGING ONLY #####################################

testing buildVisualWordList
"""

if __name__ == "__main__":
    
    x = [[3, 4], [1, 2], [5, 6], [9, 10], [7, 8], [13, 14], [11, 12], [15, 16], [17, 18], [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30], [31, 32]]
    ht = 3
    indx, offs = buildVisualWordList(x, ht)

