# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 09:14:43 2020

@author: Joshua Bensemann
"""
from csv import writer


def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)