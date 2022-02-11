"""vae class model loader from file"""

import os

from model.vae import vae


def getModels(customName=''):
    """
    Create a vae model and fill it with previosly saved weights.

    Parameters:
        customName : string
            String to append at each model's default file name
            Default ''
    """

    autoencoder_model = vae()
    autoencoder_model.load_from_file(customName)
    return autoencoder_model
