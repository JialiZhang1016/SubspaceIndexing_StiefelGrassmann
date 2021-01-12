#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 15:12:57 2021

@author: kbiren
"""

import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_digits
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import ImageGrid

import seaborn as sns
import pandas as pd

import tensorflow as tf
from scipy.spatial import Delaunay

import umap
import umap.plot

#------------------------------------------------------------------------------

def sample_points(vertices, n_samples=500):
    '''
    Parameters
    ----------
    vertices : float
        It is a (n x m) array which represents the corner points of a convex-hull.
    n_samples : int, optional
        No. of samples to generate. The default is 500.

    Returns
    -------
    P : float
        Its an array of size (n_samples x m) which is data that are generated.
        
    '''
    tri = Delaunay(vertices)
    faces = tri.simplices
    vec_cross = np.cross(vertices[faces[:, 0], :] - vertices[faces[:, 2], :],
                       vertices[faces[:, 1], :] - vertices[faces[:, 2], :])
    face_areas = np.sqrt(vec_cross ** 2)
    face_areas = face_areas / np.sum(face_areas)
    
    n_samples_per_face = np.ceil(n_samples * face_areas).astype(int)
    floor_num = np.sum(n_samples_per_face) - n_samples
    if floor_num > 0:
        indices = np.where(n_samples_per_face > 0)[0]
        floor_indices = np.random.choice(indices, floor_num, replace=True)
        n_samples_per_face[floor_indices] -= 1
    
    n_samples = np.sum(n_samples_per_face)
    
    # Create a vector that contains the face indices
    sample_face_idx = np.zeros((n_samples, ), dtype=int)
    acc = 0
    for face_idx, _n_sample in enumerate(n_samples_per_face):
        sample_face_idx[acc: acc + _n_sample] = face_idx
        acc += _n_sample
    
    r = np.random.rand(n_samples, 2)
    A = vertices[faces[sample_face_idx, 0], :]
    B = vertices[faces[sample_face_idx, 1], :]
    C = vertices[faces[sample_face_idx, 2], :]
    # barycentric coordinate for 2-simplex surface sampling
    P = (1 - np.sqrt(r[:,0:1])) * A + np.sqrt(r[:,0:1]) * (1 - r[:,1:]) * B + np.sqrt(r[:,0:1]) * r[:,1:] * C
    return P
    

if __name__ == "__main__":
       
    #==============================================================================
    #                                 cifa10
    #==============================================================================
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    #------------------------------------------------------------------------------
    n_samples = 2048    # no. of samples used for training UMAP
    n_components = 2     # reduced dimension

    data = x_train[:n_samples].reshape(-1, 32*32*3)
    labels = y_train[:n_samples].flatten()

    mapper = umap.UMAP(random_state=42, n_components=n_components, n_neighbors=200, min_dist=0.1).fit(data)
    embedding = mapper.transform(data)
    umap.plot.points(mapper, labels=labels)

    #------------------------------------------------------------------------------

    u_labels = np.unique(labels)
    cluster_mean = np.zeros((len(u_labels), n_components))
    for l in u_labels:
        cluster_mean[l, :] = np.mean(embedding[labels==l], axis=0)

    aug_points = sample_points(cluster_mean, n_samples=500)

    inv_transformed_points = mapper.inverse_transform(aug_points)
    rec_image = inv_transformed_points.reshape(-1, 32, 32, 3).astype(np.uint8)

    #------------------------------------------------------------------------------
    fig = plt.figure(figsize=(12,6))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(2, 5),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                    )

    for ax, im in zip(grid, rec_image):
        ax.imshow(im)
    plt.show()

    #==============================================================================
    #                                 mnist
    #==============================================================================
    x_train, y_train = sklearn.datasets.fetch_openml('mnist_784', version=1, return_X_y=True)

    #------------------------------------------------------------------------------
    n_samples = 2048   # no. of sample sused for training UMAP
    n_components = 2   # reduced dimension

    data = np.array(x_train)[:n_samples]
    labels = y_train[:n_samples]
    labels = np.array([int(l) for l in labels])

    mapper = umap.UMAP(random_state=42, n_components=n_components, n_neighbors=15, min_dist=0.1).fit(data)
    embedding = mapper.transform(data)
    umap.plot.points(mapper, labels=labels)

    #------------------------------------------------------------------------------

    u_labels = np.unique(labels)
    cluster_mean = np.zeros((len(u_labels), n_components))
    for l in u_labels:
        cluster_mean[l, :] = np.mean(embedding[labels==l], axis=0)
    
    aug_points = sample_points(cluster_mean, n_samples=500)

    inv_transformed_points = mapper.inverse_transform(aug_points)
    rec_image = inv_transformed_points.reshape(-1, 28, 28).astype(np.uint8)

    #------------------------------------------------------------------------------
    fig = plt.figure(figsize=(12,6))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(10, 10),  # creates 10x10 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    for ax, im in zip(grid, rec_image):
        ax.imshow(im)
    plt.show()