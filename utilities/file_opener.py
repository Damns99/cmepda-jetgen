'''File opener for the .h5 dataset'''

import os
import numpy as np
import h5py

data_path = os.path.join(os.path.dirname(__file__), '..', 'data')

filename = ['jetImage_7_100p_0_10000.h5', 'jetImage_7_100p_10000_20000.h5',
            'jetImage_7_100p_30000_40000.h5', 'jetImage_7_100p_40000_50000.h5',
            'jetImage_7_100p_50000_60000.h5', 'jetImage_7_100p_60000_70000.h5',
            'jetImage_7_100p_70000_80000.h5', 'jetImage_7_100p_80000_90000.h5']


def get_jet_list(test=False, idx=(1, 2, 3, 10, 11)):
    '''
    File opener function for the .h5 dataset.

    Parameters:
        test : bool
            If True open only the first file (use to speedup code testing)
        idx : list or tuple
            List of indexes of the jets field features to select
            Default (1, 2, 3, 10, 11)

    Returns:
        jet_list : numpy 2d array
            Each row is an array of the selected jet features
        target : numpy 2d array
            Each row is an array corresponding to the one-hot encoded jet type
    '''

    jet_list = np.array([])
    target = np.array([])

    for file in filename:
        f = h5py.File(os.path.join(data_path, file))
        features_list = np.array(f.get('jets'))
        my_jet_list = features_list[:, idx]
        my_target = features_list[:, -6:-1]
        if jet_list.size:
            jet_list = np.concatenate([jet_list, my_jet_list], axis=0)
            target = np.concatenate([target, my_target], axis=0)
        else:
            jet_list = my_jet_list
            target = my_target
        del my_jet_list, my_target
        f.close()
        if test:
            break
    return jet_list, target


def stand_data(jet_list):
    '''
    Standardization of dataset. Mean and standard deviation are computed among
    each feature column.

    Parameters:
        jet_list : numpy 2d array
            Array to standardize.

    Returns:
        Numpy array of standardized data.
    '''
    means = np.mean(jet_list, axis=0)
    stds = np.std(jet_list, axis=0)
    return (jet_list - means) / stds


def get_models(custom_name=''):
    """
    Create a vae model and fill it with previosly saved weights.

    Parameters:
        customName : string
            String to append at each model's default file name
            Default ''
    """
    from tensorflow.keras.models import load_model

    train_path = os.path.join(os.path.dirname(
        __file__), '..', 'trained_models')
    print(f'searching in {train_path}')
    return load_model(os.path.join(train_path, 'vae'))
