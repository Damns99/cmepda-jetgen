"""Use a saved vae model to generate some new jets and confront them with real data."""

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

from utilities.file_opener import getJetList
from utilities.plots import jetHist

def jet_gen(particleType, nEvents=1, seed=42):
    """
    Open a saved vae model and generate new jets of a given jet type from random noise

    Parameters:
        particleType : list or tuple
            selected jet type in one-hot encoded form.
        nEvents : int
            number of jets to generate.
            Default 1
        seed : int
            seed for the np.random random number generator, used for the noise
            Default 42

    Returns:
        A numpy 2d array of nEvents rows each with generated jet features
    """

    from utilities.model_getter import getModels

    autoencoder_model = getModels()

    np.random.seed(seed)

    particle = np.tile(particleType, (nEvents, 1))
    noise = np.random.normal(size=(nEvents, autoencoder_model.encDimensions))

    return np.array(autoencoder_model.decoder_predict(noise, particle))

with tf.device('/CPU:0'):

    jetList, target = getJetList()

    w1 = 1000
    w3 = 100

    jetList[:, 0] = jetList[:, 0] / w1
    jetList[:, 2] = jetList[:, 2] / w3
    # jetList[:, 1] = np.abs(jetList[:, 1])

    particleType = [0, 0, 0, 0, 1]
    particleTag = np.argmax(particleType)

    generated_jets = jet_gen(particleType, 10000, 789)

    print(generated_jets[:10])

    filterType = np.all((target == particleType), axis=1)
    jetList = jetList[filterType, :]
    target = target[filterType, :]

    print(jetList[:10])
    print(target[:10])

    featureNames = ['pt', 'abs(eta)', 'mass', 'tau32_b1', 'tau32_b2']
    jetHist(jetList, featureNames, customName=f'_type{particleTag}_true', bins=100)
    jetHist(generated_jets, featureNames, customName=f'_type{particleTag}_gen', bins=100)

    plt.show()
