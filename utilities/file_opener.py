'''File opener for the .h5 dataset'''

import os
import numpy as np
import h5py


data_path = os.path.join(os.path.dirname(__file__), '..', 'data')

filename = ['jetImage_7_100p_0_10000.h5', 'jetImage_7_100p_10000_20000.h5',
            'jetImage_7_100p_30000_40000.h5', 'jetImage_7_100p_40000_50000.h5',
            'jetImage_7_100p_50000_60000.h5', 'jetImage_7_100p_60000_70000.h5',
            'jetImage_7_100p_70000_80000.h5', 'jetImage_7_100p_80000_90000.h5']


def getJetList(test=False, idx=(1, 2, 3, 10, 11)):
    '''
    File opener function for the .h5 dataset.

    Parameters:
        test : bool
            If True open only the first file (use to speedup code testing)
        idx : list or tuple
            List of indexes of the jets field features to select
            Default (1, 2, 3, 10, 11)

    Returns:
        jetList : numpy 2d array
            Each row is an array of the selected jet features
        target : numpy 2d array
            Each row is an array corresponding to the one-hot encoded jet type
    '''

    jetList = np.array([])
    target = np.array([])

    for file in filename:
        f = h5py.File(os.path.join(data_path, file))
        featuresList = np.array(f.get('jets'))
        myJetList = featuresList[:, idx]
        myTarget = featuresList[:, -6:-1]
        if jetList.size:
            jetList = np.concatenate([jetList, myJetList], axis=0)
            target = np.concatenate([target, myTarget], axis=0)
        else:
            jetList = myJetList
            target = myTarget
        del myJetList, myTarget
        f.close()
        if test:
            break
    return jetList, target
