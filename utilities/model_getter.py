import os
from tensorflow.keras.models import load_model

path = os.path.join(os.path.dirname(__file__), '..', 'trained_models')


def getModels():
    encoder_model = load_model(os.path.join(path, 'encoder'))
    decoder_model = load_model(os.path.join(path, 'decoder'))
    autoencoder_model = load_model(os.path.join(path, 'autoencoder'))
    return encoder_model, decoder_model, autoencoder_model
