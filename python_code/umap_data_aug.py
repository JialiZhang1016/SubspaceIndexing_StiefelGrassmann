#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 15:12:57 2021

@author: Birendra Kathariya (UMKC) and Wenqing Hu (Missouri S&T)
"""

import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_digits
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_olivetti_faces

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
        Number of samples to generate. The default is 500.

    Returns
    -------
    P : float
        It is an array of size (n_samples x m) which is data that are generated.
        
    '''
    # Create Delauney Triangle mesh (2-simplex) surface from the mean-data
    tri = Delaunay(vertices)
    faces = tri.simplices
    vec_cross = np.cross(vertices[faces[:, 0], :] - vertices[faces[:, 2], :],
                       vertices[faces[:, 1], :] - vertices[faces[:, 2], :])
    face_areas = np.sqrt(vec_cross ** 2)
    face_areas = face_areas / np.sum(face_areas)
    
    # Sample points in the triangle mesh surface proportional to their area
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
    # Used barycentric coordinate surface sampling approach
    # Total points sampled from all the triangle-meshes are equal to the target point count to be augmented

    P = (1 - np.sqrt(r[:,0:1])) * A + np.sqrt(r[:,0:1]) * (1 - r[:,1:]) * B + np.sqrt(r[:,0:1]) * r[:,1:] * C
    
    return P


# Augmentation by UMAP and Linear Interpolation via Simplicial Approximation

def UMAP_Augmentation(data, labels, number_components, number_samples, number_neighbors):

    # Apply UMAP to reduce data dimension (n x m => n x d, d << m)
    mapper = umap.UMAP(random_state=42, n_components=number_components, n_neighbors=number_neighbors, min_dist=0.1).fit(data)
    embedding = mapper.transform(data)
    #umap.plot.points(mapper, labels=labels)

    # Cluster this dimension-reduced data based on their label
    unique_labels = np.unique(labels)
    number_labels = len(unique_labels)
    cluster_mean = np.zeros((number_labels, number_components))
    # Compute mean of the clusters (c x d, c is no. of classes )
    for l in range(number_labels):
        cluster_mean[l, :] = np.mean(embedding[labels==unique_labels[l]], axis=0)

    # Create Delauney Triangle mesh (2-simplex) surface from the mean-data
    # Sample points in the triangle mesh surface proportional to their area
    # Used barycentric coordinate surface sampling approach
    # Total points sampled from all the triangle-meshes are equal to the target point count to be augmented
    augmented_points = sample_points(cluster_mean, number_samples)

    # Apply inverse UMAP to reconstruct the data to their original dimension
    print("---------------UMAP Inverse Transform Started!---------------")
    inv_transformed_points = []
    for i in range(number_samples):
        print("UMAP Transformed Point #", i)
        inv_transformed_points.append(mapper.inverse_transform([augmented_points[i]])[0])
    inv_transformed_points = np.array(inv_transformed_points)

    return inv_transformed_points
 

    

if __name__ == "__main__":
    
    doMNIST = 0
    doCIFAR10 = 0
    doOlivetti = 1

    # load MNIST dataset
    if doMNIST:
        # load the MNIST dataset
        # structure: 
        #    x_train: list (60000, 28 , 28) dtype=unit8
        #    x_test: list (10000, 28 , 28) dtype=unit8
        #    y_train: list (60000, 1) dtype=unit8
        #    y_test: list (10000, 1) dtype=unit8
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # preprocess the dataset to fit the format we use
        data_original_train = {"x": [], "y": []}
        data_original_test = {"x": [], "y": []}
        # first turn the matrices of x_train and x_test to 28 x 28 = 784 dimensional vectors
        for i in range(60000):
            data_original_train["x"].append(np.reshape(x_train[i], 784))
            data_original_train["y"].append(y_train[i])
        for i in range(10000):
            data_original_test["x"].append(np.reshape(x_test[i], 784))
            data_original_test["y"].append(y_test[i])
    # load CIFAR10 dataset
    if doCIFAR10:
        # load the CIFAR-10 dataset
        # structure: 
        #    x_train: list (50000, 32 , 32, 3) dtype=unit8
        #    x_test: list (10000, 32 , 32, 3) dtype=unit8
        #    y_train: list (50000, 1) dtype=unit8
        #    y_test: list (10000, 1) dtype=unit8
        cifar10 = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # preprocess the dataset to fit the format we use
        data_original_train = {"x": [], "y": []}
        data_original_test = {"x": [], "y": []}
        # first turn the matrices of x_train and x_test to 32 x 32 x 3 = 3072 dimensional vectors
        for i in range(50000):
            data_original_train["x"].append(np.reshape(x_train[i], 3072))
            data_original_train["y"].append(y_train[i][0])
        for i in range(10000):
            data_original_test["x"].append(np.reshape(x_test[i], 3072))
            data_original_test["y"].append(y_test[i][0])
    if doOlivetti:
        # load the ATT Olivetti dataset
        # structure: 
        #    images: list (400, 4096) dtype=float32
        #    target: list (400, 0) dtype=float32
        OlivettiFaceData = sklearn.datasets.fetch_olivetti_faces(random_state=0)
        x = OlivettiFaceData.images
        y = OlivettiFaceData.target
        # preprocess the dataset to fit the format we use
        data_original_train = {"x": [], "y": []}
        data_original_test = {"x": [], "y": []}
        # extract the training and testing data sets
        indexes = np.arange(400) 
        np.random.shuffle(indexes)
        for i in range(350):
            data_original_train["x"].append(np.reshape(x[indexes[i]], 4096))
            data_original_train["y"].append(y[indexes[i]])
        for i in range(50):
            data_original_test["x"].append(np.reshape(x[indexes[350+i]], 4096))
            data_original_test["y"].append(y[indexes[350+i]])


    data = np.array(data_original_train["x"])[:350]
    labels = np.array(data_original_train["y"])[:350]

    number_samples = 500
    number_components = 2
    number_neighbors = 20

    inv_transformed_points = UMAP_Augmentation(data, labels, number_components, number_samples, number_neighbors)
    
    #------------------------------------------------------------------------------
    if doMNIST:
        rec_image = inv_transformed_points.reshape(-1, 28, 28).astype(np.uint8)
    elif doCIFAR10:
        rec_image = inv_transformed_points.reshape(-1, 32, 32, 3).astype(np.uint8)
    elif doOlivetti:
        rec_image = inv_transformed_points.reshape(-1, 64, 64).astype(np.float32)
    else:
        print("No data set chosen!\n")

    fig = plt.figure(figsize=(12,6))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(10, 10),  # creates 10x10 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    for ax, im in zip(grid, rec_image):
        ax.imshow(im)
    plt.show()