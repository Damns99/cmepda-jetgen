"""Open the dataset, evaluate the model and confront results with expectations."""

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

# Sfrom sklearn.cluster import Birch

from utilities.file_opener import getJetList, standData
from utilities.model_getter import getModels
from utilities.plots import jetScatter, jetScatter3D
from utilities.figure_saver import saveFig


with tf.device('/CPU:0'):
    jetList, target = getJetList()

    # jetList = jetList[:1000, :]
    # target = target[:1000, :]

    njets = jetList.shape[0]

    autoencoder_model = getModels()

    jetTag = np.argmax(target, axis=1)

    jetList = standData(jetList)

    encoded_features = autoencoder_model.encoder_predict(jetList, target)
    decoded_jets = autoencoder_model.decoder_predict(encoded_features, target)

    print(decoded_jets[:10, :])
    print(jetList[:10, :])

    plt.figure()
    plt.subplot(121)
    jetScatter(encoded_features, jetTag, 0, 1)
    plt.subplot(122)
    jetScatter(encoded_features, jetTag, 0, 2)
    saveFig("encoded_features_2d")

    plt.figure()
    jetScatter3D(encoded_features, jetTag)
    saveFig("encoded_features_3d")

    # brc = Birch(branching_factor=50, n_clusters=5, threshold=0.1)
    # brc.fit(encoded_features)
    # predTag = brc.predict(encoded_features)

    # plt.figure()
    # plt.subplot(121)
    # jetScatter(encoded_features, predTag, 0, 1)
    # plt.subplot(122)
    # jetScatter(encoded_features, predTag, 0, 2)
    # saveFig("birch_clusters_2d")
    #
    # plt.figure()
    # jetScatter3D(encoded_features, predTag)
    # saveFig("birch_clusters_3d")

    for i in range(5):
        particleType = np.zeros(5)
        particleType[i] = 1
        filterType = np.all((target == particleType), axis=1)
        encoded_features_filtered = encoded_features[filterType, :]
        jetTag_filtered = jetTag[filterType]
        plt.figure()
        jetScatter3D(encoded_features_filtered, jetTag_filtered)
        saveFig(f"encoded_features_3d_type{i}")

    plt.show()
