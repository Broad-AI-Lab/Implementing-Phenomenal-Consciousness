# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 17:22:09 2020

@author: qazca
"""
import librosa
from data_processing.utilities import concat_arrays

#Extract Mel Spectrogram from a sound file
def extract_melspectrogram(filename):
    print(filename, end="...")
    mel_spectrogram = None
    
    try:
        y, sr = librosa.load(filename)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        new_shape = [1]
        current_shape = mel_spectrogram.shape
        
        for i in range(len(current_shape)):
            new_shape.append(current_shape[i])
            
        new_shape = tuple(new_shape)
        mel_spectrogram = mel_spectrogram.reshape(new_shape)
        
    except Exception as e:
        print("File {} caused the following exception: {}".format(filename, e))
    
 
    print("Done!")
    
    return mel_spectrogram


#Convert list of filenames into single dataset
def create_dataset(file_list):
    processed_sound_list = []    
    
    for file in file_list:
        processed_sound = extract_melspectrogram(file)
        
        if file is not None:
            processed_sound_list.append(processed_sound)
            
    dataset = concat_arrays(processed_sound_list)
    
    return dataset
    