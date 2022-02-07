import os

from model.vae import vae

path = os.path.join(os.path.dirname(__file__), '..', 'trained_models')


def getModels(customName=''):
    autoencoder_model = vae()
    autoencoder_model.load_from_file(customName)
    return autoencoder_model
