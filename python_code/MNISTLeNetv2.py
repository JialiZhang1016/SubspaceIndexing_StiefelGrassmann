#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 19:33:46 2021

https://github.com/guptajay/Kaggle-Digit-Recognizer
@author: kbiren
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import pandas as pd



if __name__ == "__main__":
    
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Padding the images by 2 pixels since in the paper input images were 32x32
    x_test = np.pad( x_test[:,:,:, np.newaxis], ((0,0),(2,2),(2,2),(0,0)), 'constant')
    mean_px = x_test.mean().astype(np.float32)
    std_px = x_test.std().astype(np.float32)
    x_test = (x_test - mean_px)/(std_px)
    #--------------------------------------------------------------------------
    
    LeNet5Model = load_model('LeNetv2.h5')
    y_pred = LeNet5Model.predict(x_test)
    y_pred = np.argmax(y_pred,axis = 1)
    accuracy = np.sum(y_test == y_pred) / len(y_test);
    acc = 100*accuracy;
    print('Test accuracy:', acc)
    
    #--------------------------------------------------------------------------
    
    # LeNet5Model = load_model('LeNetv2.h5')
    # LeNet5Model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # y_test = to_categorical(y_test, num_classes = 10)
    # score, acc = LeNet5Model.evaluate(x_test, y_test, batch_size=128)
    # print('Test score:', score)
    # print('Test accuracy:', acc)
    
    #--------------------------------------------------------------------------
    
