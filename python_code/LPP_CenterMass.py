#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 17:32:08 2020

%%%%%%%%%%%%%%%%%%%% LPP analysis based on Grassmann center of mass calculation %%%%%%%%%%%%%%%%%%%%

@author: Wenqing Hu (Missouri S&T)
"""

from Stiefel_and_Grassmann_Optimization import Stiefel_Optimization
from buildVisualWordList import buildVisualWordList

x = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30], [31, 32]]
ht = 2
indx, leafs, mbrs = buildVisualWordList(x, ht)
print("leafs=", leafs)
print("indx=", indx)
print("mbrs=", mbrs)
