# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 09:28:48 2020

@author: Joshua Bensemann
"""
import tensorflow as tf
import numpy as np


#loading data from numpy zip files
def load_data(filepath):
    data = np.load(filepath)

    X = data["arr_0"]
    y = data["arr_1"] 
    
    return X, y


def load_split_data(filepath):
    data = np.load(filepath)

    x_train = data["arr_0"]
    y_train = data["arr_1"]
    x_test = data["arr_2"]
    y_test = data["arr_3"]

    assert x_train.shape[0] == y_train.shape[0] and x_test.shape[0] == y_test.shape[0]

    return x_train, y_train, x_test, y_test


#apply pooling to numpy data
def pool_data(data, pool_type="max", pool_size=4, batches=10000):
    pool = None
    arrays = []
    
    num_batches = int(data.shape[0] / batches)
    
    for i in range (num_batches+1):
        start_point = i * batches
        end_point = (i+1) * batches
        
        if i==num_batches:
            end_point = data.shape[0]
            
        batch = data[start_point:end_point]
    
        if pool_type == "average":
            pool = tf.keras.layers.AveragePooling2D(pool_size=pool_size, dtype='float32')  
        elif pool_type == "max":
            pool = tf.keras.layers.MaxPooling2D(pool_size=pool_size, dtype='float32')  

        arrays.append(pool(batch).numpy())
    
    pooled_data = np.concatenate(arrays)
    
    return pooled_data