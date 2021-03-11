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
    #print("x=", x)
    #print("ht=", ht)
    # var
    n = len(x)
    #print("n=", n)
    kd = len(x[0])
    nNode = 2**(ht+1) - 1 
    nLeafNode = 2**ht
        
    # intermediate storages, offs is a list of lists
    offs = [[] for _ in range(nNode)]
    #print("offs=", offs)
    
    # initialize indx as a dictionary {"d_cuts": [store cut dimensions], "v_cuts": [store cut values]}
    indx = {"d_cuts": [], "v_cuts": []}
    
    # first cut
    # cut at dimension 0
    indx['d_cuts'].append(0) 
    # sort the x-value at first dimension, sv, soffs are lists
    soffs, sv = zip(*sorted(enumerate(x.T[0]), key=itemgetter(1))) 
    sv = list(sv)
    soffs = list(soffs)
    #print("sv=", sv)
    #print("soffs=", soffs)
    indx['v_cuts'].append(sv[math.floor(n/2)-1])
    #print("indx=", indx)
    
    # split at mv, left child x(:,1) <= mv, right child x(:,1) > mv
    offs[1]=soffs[1-1 : math.floor(n/2)] 
    offs[2]=soffs[math.floor(n/2)+1-1 : n]
    #print("offs=", offs)
    
    for h in range(ht-1):
        #print("h=", h)
        # parents nodes at height h+1
        for k in range(2**(h+1), 2**(h+1+1)):
            #print("k=", k)
            # compute covariance
            offs_k = offs[k-1]
            #print("offs_k=", offs_k)
            nk = len(offs_k)
            # median offs 
            moffs = math.floor(nk/2)-1 
            # pick all rows of x with indexes from offs_k
            x_offs_k = [x[_] for _ in offs_k]
            x_offs_k = np.array(x_offs_k)
            #print("x_offs_k=", x_offs_k)
            # compute the covariance of x in offs_k along the kd dimensions, find the dimension with the maximal variance
            sk = np.var(x_offs_k, 0)
            #print("sk=", sk)
            max_s = max(sk)
            sk = list(sk)
            d_cut = sk.index(max(sk))
            #print("d_cut=", d_cut)
            # cut x in offs_k at dimension d_cut
            indx['d_cuts'].append(d_cut) 
            # sort the x-value in offs_k at dimension d_cut, sv, soffs are lists
            soffs, sv = zip(*sorted(enumerate(x_offs_k.T[d_cut]), key=itemgetter(1))) 
            sv = list(sv)
            soffs = list(soffs)
            #print("sv=", sv)
            #print("soffs=", soffs)
            indx['v_cuts'].append(sv[moffs])
            #print("indx=", indx)
            # current parent node k, left kid would be 2k, right kid would be 2k+1
            offs[2*k-1]     = [offs_k[_] for _ in soffs[1-1:moffs+1]]
            offs[2*k+1-1]   = [offs_k[_] for _ in soffs[moffs+1+1-1:nk]] 
            
            # prompt
            print("split [", k+1, ":", nk, "] at", d_cut, ": ", sv[moffs]) 
                
            # clean up node k
            offs[k-1] = []
            
    #print("offs=", offs)
    # leaf nodes
    leafs = sorted([offs[2**ht+j-1] for j in range(nLeafNode)])
    #print("leafs=", leafs)
    
    mbrs = [{"min": [min([x[_][col] for _ in leafs[j]]) for col in range(kd)], "max": [max([x[_][col] for _ in leafs[j]]) for col in range(kd)]} for j in range(nLeafNode)]
    #print("mbrs=", mbrs)
    
    return indx, leafs, mbrs





"""
################################ MAIN TESTING FILE #####################################
################################ FOR DEBUGGING ONLY #####################################

testing buildVisualWordList
"""

if __name__ == "__main__":
    
    x = [[2, 1], [4, 3], [5, 6], [8, 7], [9, 10], [12, 11], [13, 14], [16, 15], [17, 18], [20, 19], [21, 22], [24, 23], [25, 26], [28, 27], [30, 29], [31, 32]]
    ht = 3
    indx, leafs, mbrs = buildVisualWordList(x, ht)
    print("leafs=", leafs)
    print("indx=", indx)
    print("mbrs=", mbrs)
