# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 16:04:40 2020

@author: Joshua Bensemann
"""
import numpy as np


#Detects the largest array size of all arrays
def determine_largest_array(array_list):
    print("Largest arrary is:", end="")
    num_dimensions = len(array_list[0].shape)
    dimension_sizes = np.zeros(num_dimensions)
    
    for array in array_list:
        array_shape = array.shape
        
        for i in range(num_dimensions):
            
            if array_shape[i] > dimension_sizes[i]:
                dimension_sizes[i] = array_shape[i]
    
    size = dimension_sizes.tolist()
    size = tuple(size)
    print(size)    
        
    return size
          

#Automatically detect difference from largest array and apply padding      
def pad_array(array, largest_array, method="after"):
    padding_size = None
    
    if method == "after":
        padding_size_list = [(0, int(largest_array[i]-array.shape[i])) for i in range(len(largest_array))]
        padding_size = tuple(padding_size_list)
        
    padded_array = np.pad(array, padding_size)
    
    assert padded_array.shape == largest_array
    
    return padded_array
    

#Takes list of numpy arrays, pads them and then concatenates
def concat_arrays(array_list):
    largest_array = determine_largest_array(array_list)
    
    for i, array in enumerate(array_list):
        print("Checking and padding array {}...".format(i), end="")
        
        if array.shape != largest_array:
            array_list[i] = pad_array(array, largest_array)
            print("padded...", end="")
            
        print("Done!")
            
    final_array = np.concatenate(array_list)


    print("Full array creation done! Shape:{}".format(final_array.shape))
    return final_array