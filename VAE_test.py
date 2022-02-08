import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

from sklearn.cluster import Birch

from utilities.file_opener import getJetList
from utilities.model_getter import getModels
from utilities.plots import historyPlot, jetScatter, jetHist2D
from model.vae import vae

with tf.device('/CPU:0'):
    jetList, target = getJetList()

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

    plt.show()
