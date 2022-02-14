"""Open the dataset, create a vae model and train it, saving history and weights."""

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

from utilities.file_opener import getJetList
from utilities.plots import historyPlot
from utilities.figure_saver import saveFig
from model.vae import vae

w1 = 1000
w3 = 100

with tf.device('/GPU:0'):
    jetList, target = getJetList()

    jetList[:, 0] = jetList[:, 0] / w1
    jetList[:, 2] = jetList[:, 2] / w3
    jetList[:, 1] = np.abs(jetList[:, 1])

    autoencoderModel = vae()
    autoencoderModel.summary()

    lossWeights = [1.0, 0.1]
    learningRate = 0.01

    autoencoderModel.compile(lossWeights=lossWeights,
                             learningRate=learningRate)

    validationSplit = 0.5
    batchSize = 400
    epochs = 30

    history = autoencoderModel.fit(jetList, target, validationSplit=validationSplit,
                                   batchSize=batchSize, epochs=epochs)

    plt.figure()
    historyPlot(history)
    saveFig('history')

    autoencoderModel.save()
    autoencoderModel.save_weights()
    plt.show()
