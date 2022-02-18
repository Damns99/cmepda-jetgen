"""Jet generator module"""

import numpy as np
from utilities.file_opener import get_models

def jet_gen(particle_type, n_events=1, seed=42):
    """
    Open a saved vae model and generate new jets of a given jet type from random noise

    Parameters:
        particle_type : list or tuple
            selected jet type in one-hot encoded form.
        n_events : int
            number of jets to generate.
            Default 1
        seed : int
            seed for the np.random random number generator, used for the noise
            Default 42

    Returns:
        A numpy 2d array of nEvents rows each with generated jet features
    """

    autoencoder_model = get_models()

    np.random.seed(seed)

    particle = np.tile(particle_type, (n_events, 1))
    noise = np.random.normal(size=(n_events, autoencoder_model.enc_dimensions))

    return np.array(autoencoder_model.decoder_predict(noise, particle))
