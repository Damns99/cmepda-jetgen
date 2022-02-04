'''File opener function for the .h5 dataset'''

import os

import numpy as np
import h5py


data_path = os.path.join(os.path.dirname(__file__), '..', 'data')

filename = ['jetImage_7_100p_0_10000.h5', 'jetImage_7_100p_10000_20000.h5',
            'jetImage_7_100p_30000_40000.h5', 'jetImage_7_100p_40000_50000.h5',
            'jetImage_7_100p_50000_60000.h5', 'jetImage_7_100p_60000_70000.h5',
            'jetImage_7_100p_70000_80000.h5', 'jetImage_7_100p_80000_90000.h5']


def getJetList(test=False, idx=[5, 8, 11]):
    '''File opener function for the .h5 dataset.
    '''
    jetList = np.array([])
    target = np.array([])

    for file in filename:
        f = h5py.File(os.path.join(data_path, file))
        # for pT, etarel, phirel [5, 8, 11]
        myJetList = np.array(f.get("jetConstituentList")[:, :, idx])
        myTarget = np.array(f.get('jets')[:, -6:-1])
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