"""
Created on 22/05/2020

@author: Joshua Bensemann
"""
from training import C3_Base as Base
from training import C3_Conscious as Conscious
from datasets.datasets import get_emnist_image_plus_sounds

RUNS = 30

def main():
    dataset_dict = get_emnist_image_plus_sounds(return_combined_dataset=False, superclasses=True, balance_classes=True,
                                                balance=False, relabel=True)

    for i in range(1, RUNS+1):
        print('Run number {}'.format(i))
        Base.main(i, dataset_dict, print_messages=False)
        Conscious.main(i, dataset_dict, print_messages=False)

if __name__ == '__main__':
    main()
