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
    
    