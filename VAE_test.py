import os.path

import h5py
import numpy as np
from matplotlib import pyplot as plt

from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Lambda, Reshape
from tensorflow.keras.models import Model
import tensorflow as tf

from sklearn.cluster import Birch

from utilities.file_opener import getJetList
from utilities.model_getter import getModels
from utilities.plots import historyPlot, jetScatter, jetHist2D
from model.vae import vae

with tf.device('/CPU:0'):
    jetList, target = getJetList()

    jet_shape = jetList.shape[1:]

    autoencoder_model = getModels()

    jetTag = np.argmax(target, axis=1)

    encoded_features = autoencoder_model.encoder_predict(jetList=jetList)
    decoded_jets = autoencoder_model.decoder_predict(encoded_features)

    plt.figure()
    jetScatter(encoded_features, jetTag)

    brc = Birch(branching_factor=50, n_clusters=5, threshold=0.1)
    brc.fit(encoded_features)
    predTag = brc.predict(encoded_features)

    plt.figure()
    jetScatter(encoded_features, predTag)

    plt.figure()
    jetHist2D(jetList[jet_index, :, :])

    plt.figure()
    jetHist2D(decoded_jets[jet_index, :, :])

    # tmpx = np.arange(jet_shape[0]+1)
    # tmpy = np.arange(jet_shape[1]+1)
    # tmpx, tmpy = np.meshgrid(tmpx, tmpy)
    #
    # plt.figure(3)
    # #plt.imshow(jetList[jet_index, :, :], origin='lower', cmap='viridis_r',
    # #           extent=[0, jet_shape[0], 0, jet_shape[1]])
    # plt.pcolormesh(tmpx, tmpy, jetList[jet_index, :, :], cmap='viridis_r', shading='flat')
    # plt.colorbar()
    #
    # plt.figure(4)
    # #plt.imshow(decoded_jets[jet_index, :, :], origin='lower', cmap='viridis_r',
    # #           extent=[0, jet_shape[0], 0, jet_shape[1]])
    # plt.pcolormesh(tmpx, tmpy, decoded_jets[jet_index, :, :], cmap='viridis_r', shading='flat')
    # plt.colorbar()

    plt.show()
