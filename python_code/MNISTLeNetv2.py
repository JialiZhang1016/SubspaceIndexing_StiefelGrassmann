#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 07:03:16 2021

@author: kbiren
"""

import numpy as np
import tensorflow as tf
import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation, BatchNormalization
from keras.models import Model, Sequential, load_model
#from keras.utils import to_categorical
from keras.regularizers import l2



#------------------------------------------------------------------------------

def LeNet5v2(input_shape = (32, 32, 1), classes = 10):
    """
    Implementation of a modified LeNet-5.
    Only those layers with learnable parameters are counted in the layer numbering.
    
    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    model = Sequential([
        
    # Layer 1
    Conv2D(filters = 32, kernel_size = 5, strides = 1, activation = 'relu', input_shape = (32,32,1), kernel_regularizer=l2(0.0005), name = 'convolution_1'),
    
    # Layer 2
    Conv2D(filters = 32, kernel_size = 5, strides = 1, name = 'convolution_2', use_bias=False),
    
    # Layer 3    
    BatchNormalization(name = 'batchnorm_1'),
        
    # -------------------------------- #  
    Activation("relu"),
    MaxPooling2D(pool_size = 2, strides = 2, name = 'max_pool_1'),
    Dropout(0.25, name = 'dropout_1'),
    # -------------------------------- #  
        
    # Layer 3
    Conv2D(filters = 64, kernel_size = 3, strides = 1, activation = 'relu', kernel_regularizer=l2(0.0005), name = 'convolution_3'),
        
    # Layer 4
    Conv2D(filters = 64, kernel_size = 3, strides = 1, name = 'convolution_4', use_bias=False),
        
    # Layer 5
    BatchNormalization(name = 'batchnorm_2'),
        
    # -------------------------------- #  
    Activation("relu"),
    MaxPooling2D(pool_size = 2, strides = 2, name = 'max_pool_2'),
    Dropout(0.25, name = 'dropout_2'),
    Flatten(name = 'flatten'),
    # -------------------------------- #  
        
    # Layer 6
    Dense(units = 256, name = 'fully_connected_1', use_bias=False),
        
    # Layer 7
    BatchNormalization(name = 'batchnorm_3'),
    
    # -------------------------------- #  
    Activation("relu"),
    # -------------------------------- #  
        
    # Layer 8
    Dense(units = 128, name = 'fully_connected_2', use_bias=False),
        
    # Layer 9
    BatchNormalization(name = 'batchnorm_4'),
        
    # -------------------------------- #  
    Activation("relu"),
    # -------------------------------- #  
        
    # Layer 10
    Dense(units = 84, name = 'fully_connected_3', use_bias=False),
        
    # Layer 11
    BatchNormalization(name = 'batchnorm_5'),
        
    # -------------------------------- #  
    Activation("relu"),
    Dropout(0.25, name = 'dropout_3'),
    # -------------------------------- #  

    # Output
    Dense(units = 10, activation = 'softmax', name = 'output')
        
    ])
    
    model._name = 'LeNet5v2'

    return model

def MNISTLeNetv2():
    LeNet5Model = LeNet5v2(input_shape = (32, 32, 1), classes = 10)
    #LeNet5Model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    LeNet5Model.load_weights('MNISTLeNetv2.h5')
    return LeNet5Model

#------------------------------------------------------------------------------

if __name__ == "__main__":
    
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #--------------------------------------------------------------------------
    
    # Padding the images by 2 pixels since in the paper input images were 32x32
    x_test = np.pad( x_test[:,:,:, np.newaxis], ((0,0),(2,2),(2,2),(0,0)), 'constant')
    mean_px = x_test.mean().astype(np.float32)
    std_px = x_test.std().astype(np.float32)
    x_test = (x_test - mean_px)/(std_px)
    
    
    LeNet5Model = LeNet5v2(input_shape = (32, 32, 1), classes = 10)
    #LeNet5Model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    LeNet5Model.load_weights('MNISTLeNetv2.h5')
    
    y_pred = LeNet5Model.predict(x_test)
    y_pred = np.argmax(y_pred,axis = 1)
    accuracy = np.sum(y_test == y_pred) / len(y_test);
    acc = 100*accuracy;
    print('Test accuracy:', acc)



