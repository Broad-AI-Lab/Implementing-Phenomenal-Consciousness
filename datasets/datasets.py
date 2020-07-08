# -*- coding: utf-8 -*-
"""
Created on 21/05/2020

@author: Joshua Bensemann
"""
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd
from datasets.utilities import load_data, pool_data, load_split_data
from emnist import extract_training_samples, extract_test_samples

#load mnist_images
def get_emnist_images(formatted=True, dataset='balanced'):
    """

    :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
    """
    emnist_x_train, emnist_y_train = extract_training_samples(dataset)
    emnist_x_test, emnist_y_test = extract_test_samples(dataset)

    if formatted:
        emnist_x_train = emnist_x_train / 255.0
        emnist_x_test = emnist_x_test / 255.0

        emnist_x_train = emnist_x_train.reshape((-1, 28, 28, 1))
        emnist_x_test = emnist_x_test.reshape((-1, 28, 28, 1))
        
    return emnist_x_train, emnist_y_train, emnist_x_test, emnist_y_test


def get_isolet_images():
    isolet_x_train, isolet_y_train, isolet_x_test, isolet_y_test = load_split_data('datasets/isolet_encoded.npz')

    return isolet_x_train, isolet_y_train, isolet_x_test, isolet_y_test


# load all_sounds
def get_all_sounds(formatted=True, test_size=None, **kwargs):
    filepath = "datasets/full_sound.npz"
    x, y = load_data(filepath)

    if formatted:
        sound_x_train, sound_x_test, sound_y_train, sound_y_test = format_full_sound_dataset(x, y, test_size=test_size,
                                                                                         **kwargs)

        return sound_x_train, sound_y_train, sound_x_test, sound_y_test

    else:
        return x, y


#formatting sound dataset
def format_full_sound_dataset(x, y,
                          test_size=None,
                          padding=((0, 0), (6, 6), (6, 6)),
                          target_shape=(-1, 140, 56, 1),
                          pool_type="max",
                          pool_size=(5, 2)):
    
    x = x / float(np.max(x))
    
    if padding is not None:
        x = np.pad(x, padding)
    
    if target_shape is not None:
        x = x.reshape(target_shape)

    if pool_type is not None:
        x = pool_data(x, pool_type=pool_type, pool_size=pool_size)

    if test_size is not None:
        return train_test_split(x, y, test_size=test_size, random_state=1986)
    else:
        return x, y


#load sound and images dataset   
def get_emnist_image_plus_sounds(test_size=0.2, verbose=False, return_as_numpy=False, return_combined_dataset=True, balance_superclass=False,
                                 balance_test=False, balance=True, balance_classes=False, superclasses=False, relabel=False):
    data_dict = {}

    #load emist image dataset
    image_x_train, image_y_train, image_x_test, image_y_test = get_emnist_images()
    image_x_train = image_x_train.astype('float32')
    image_y_train = image_y_train.astype('float32')
    image_x_test = image_x_test.astype('float32')
    image_y_test = image_y_test.astype('float32')

    if verbose:
        print("Images loaded successfully")

    #load sounds as images and split by test_size
    sound_x_train, sound_y_train, sound_x_test, sound_y_test = get_all_sounds(test_size=test_size)
    sound_x_train = sound_x_train.astype('float32')
    sound_x_test = sound_x_test.astype('float32')
    sound_y_train = sound_y_train.astype('float32')
    sound_y_test = sound_y_test.astype('float32')
    
    if verbose:
        print("Sounds loaded successfully")
        
    if balance_classes:
        num_sound_labels = np.max(sound_y_train) + 1
        image_x_train = image_x_train[image_y_train < num_sound_labels]
        image_y_train = image_y_train[image_y_train < num_sound_labels]    
        image_x_test = image_x_test[image_y_test < num_sound_labels]
        image_y_test = image_y_test[image_y_test < num_sound_labels]
        
    else:
        relabel = True

    if relabel:
        num_image_labels = np.max(image_y_train) + 1
        sound_y_train = sound_y_train + num_image_labels
        sound_y_test = sound_y_test + num_image_labels
        
    if superclasses:
        image_y_train_len = image_y_train.shape[0]
        image_y_test_len = image_y_test.shape[0]
        sound_y_train_len = sound_y_train.shape[0]
        sound_y_test_len = sound_y_test.shape[0]

        image_y_train = image_y_train.reshape(-1, 1)
        image_y_train = np.concatenate([image_y_train, np.ones((image_y_train_len,1))*(-1)], axis=1)

        image_y_test = image_y_test.reshape(-1, 1)
        image_y_test = np.concatenate([image_y_test, np.ones((image_y_test_len,1))*(-1)], axis=1)

        sound_y_train = sound_y_train.reshape(-1, 1)
        sound_y_train = np.concatenate([sound_y_train, np.ones((sound_y_train_len,1))], axis=1)

        sound_y_test = sound_y_test.reshape(-1, 1)
        sound_y_test = np.concatenate([sound_y_test, np.ones((sound_y_test_len,1))], axis=1)

    #set length of image datasets to be same as sound
    if balance or balance_superclass:
        image_x_train = image_x_train[:sound_x_train.shape[0]]
        image_y_train = image_y_train[:sound_y_train.shape[0]]
        image_x_test = image_x_test[:sound_x_test.shape[0]]
        image_y_test = image_y_test[:sound_y_test.shape[0]]
        
    if balance_test:
        image_x_test = image_x_test[:sound_x_test.shape[0]]
        image_y_test = image_y_test[:sound_y_test.shape[0]]



    x_train = np.concatenate([image_x_train, sound_x_train])
    y_train = np.concatenate([image_y_train, sound_y_train])
    x_test = np.concatenate([image_x_test, sound_x_test])
    y_test = np.concatenate([image_y_test, sound_y_test])

    if return_as_numpy:
        if return_combined_dataset:
            data_dict['train_dataset'] = (x_train, y_train)
            data_dict['test_dataset'] = (x_test, y_test)

        else:
            data_dict['image_train'] = (image_x_train, image_y_train)
            data_dict['image_test'] = (image_x_test, image_y_test)
            data_dict['sound_train'] = (sound_x_train, sound_y_train)
            data_dict['sound_test'] = (sound_x_test, sound_y_test)

    else:
        if return_combined_dataset:
            data_dict['train_dataset'] = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            data_dict['test_dataset'] = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        else:
            data_dict['image_train'] = tf.data.Dataset.from_tensor_slices((image_x_train, image_y_train))
            data_dict['image_test'] = tf.data.Dataset.from_tensor_slices((image_x_test, image_y_test))
            data_dict['sound_train'] = tf.data.Dataset.from_tensor_slices((sound_x_train, sound_y_train))
            data_dict['sound_test'] = tf.data.Dataset.from_tensor_slices((sound_x_test, sound_y_test))
    
    if verbose:
        print("Dataset created successfully")

    return data_dict


def get_emnist_plus_isolet(verbose=False, as_image=True, return_as_numpy=False, return_combined_dataset=False,
                           balance=False, superclasses=False):
    data_dict = {}

    if as_image:
        image_x_train, image_y_train, image_x_test, image_y_test = get_emnist_images(dataset='letters')
        sound_x_train, sound_y_train, sound_x_test, sound_y_test = get_isolet_images()

    else:
        emnist_x_train = pd.read_csv('datasets/emist_encoded_train.csv', index_col=0)
        emnist_x_test = pd.read_csv('datasets/emist_encoded_test.csv', index_col=0)
        image_y_train = emnist_x_train.pop('label').values
        image_y_test = emnist_x_test.pop('label').values
        image_x_train = emnist_x_train.values
        image_x_test = emnist_x_test.values

        isolet_x_train = pd.read_csv('datasets/isolet1+2+3+4.data', header=None)
        isolet_x_test = pd.read_csv('datasets/isolet5.data', header=None)
        last_col = isolet_x_train.columns[-1]
        sound_y_train = isolet_x_train.pop(last_col).values
        sound_y_test = isolet_x_test.pop(last_col).values
        sound_x_train = isolet_x_train.values
        sound_x_test = isolet_x_test.values

    image_x_train = image_x_train.astype('float32')
    image_y_train = image_y_train.astype('float32') - 1
    image_x_test = image_x_test.astype('float32')
    image_y_test = image_y_test.astype('float32') - 1

    sound_x_train = sound_x_train.astype('float32')
    sound_x_test = sound_x_test.astype('float32')
    sound_y_train = sound_y_train.astype('float32') - 1
    sound_y_test = sound_y_test.astype('float32') - 1

    if superclasses:
        image_y_train_len = image_y_train.shape[0]
        image_y_test_len = image_y_test.shape[0]
        sound_y_train_len = sound_y_train.shape[0]
        sound_y_test_len = sound_y_test.shape[0]

        image_y_train = image_y_train.reshape(-1, 1)
        image_y_train = np.concatenate([image_y_train, np.ones((image_y_train_len,1))*(-1)], axis=1)

        image_y_test = image_y_test.reshape(-1, 1)
        image_y_test = np.concatenate([image_y_test, np.ones((image_y_test_len,1))*(-1)], axis=1)

        sound_y_train = sound_y_train.reshape(-1, 1)
        sound_y_train = np.concatenate([sound_y_train, np.ones((sound_y_train_len,1))], axis=1)

        sound_y_test = sound_y_test.reshape(-1, 1)
        sound_y_test = np.concatenate([sound_y_test, np.ones((sound_y_test_len,1))], axis=1)

    if balance:
        image_x_train = image_x_train[:sound_x_train.shape[0]]
        image_y_train = image_y_train[:sound_y_train.shape[0]]
        image_x_test = image_x_test[:sound_x_test.shape[0]]
        image_y_test = image_y_test[:sound_y_test.shape[0]]

    x_train = np.concatenate([image_x_train, sound_x_train])
    y_train = np.concatenate([image_y_train, sound_y_train])
    x_test = np.concatenate([image_x_test, sound_x_test])
    y_test = np.concatenate([image_y_test, sound_y_test])

    if return_as_numpy:
        if return_combined_dataset:
            data_dict['train_dataset'] = (x_train, y_train)
            data_dict['test_dataset'] = (x_test, y_test)

        else:
            data_dict['image_train'] = (image_x_train, image_y_train)
            data_dict['image_test'] = (image_x_test, image_y_test)
            data_dict['sound_train'] = (sound_x_train, sound_y_train)
            data_dict['sound_test'] = (sound_x_test, sound_y_test)

    else:
        if return_combined_dataset:
            data_dict['train_dataset'] = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            data_dict['test_dataset'] = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        else:
            data_dict['image_train'] = tf.data.Dataset.from_tensor_slices((image_x_train, image_y_train))
            data_dict['image_test'] = tf.data.Dataset.from_tensor_slices((image_x_test, image_y_test))
            data_dict['sound_train'] = tf.data.Dataset.from_tensor_slices((sound_x_train, sound_y_train))
            data_dict['sound_test'] = tf.data.Dataset.from_tensor_slices((sound_x_test, sound_y_test))

    if verbose:
        print("Dataset created successfully")

    return data_dict
