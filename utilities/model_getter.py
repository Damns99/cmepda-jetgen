"""vae class model loader from file"""

import os

from tensorflow.keras.models import load_model

from model.vae import vae

def getModels(customName=''):
    """
    Create a vae model and fill it with previosly saved weights.

    Parameters:
        customName : string
            String to append at each model's default file name
            Default ''
    """

    trainPath = os.path.join(os.path.dirname(__file__), '..', 'trained_models')
    print(f'searching in {trainPath}')
    return load_model(os.path.join(trainPath, 'vae'))
