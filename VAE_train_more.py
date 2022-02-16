"""Open the dataset, load a vae model previously saved and train it more."""

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

from utilities.file_opener import getJetList
from utilities.model_getter import getModels
from utilities.plots import historyPlot
from utilities.figure_saver import saveFig

w1 = 1000
w3 = 100

with tf.device('/GPU:0'):
    jetList, target = getJetList()

    jetList[:, 0] = jetList[:, 0] / w1
    jetList[:, 2] = jetList[:, 2] / w3
    jetList[:, 1] = np.abs(jetList[:, 1])

    autoencoderModel = getModels()

    lossWeights = [1.0, 0.001, 1.0]
    learningRate = 0.005

    autoencoderModel.compile(lossWeights=lossWeights,
                             learningRate=learningRate)

    validationSplit = 0.5
    batchSize = 400
    epochs = 100

    history = autoencoderModel.fit(jetList, target, validationSplit=validationSplit,
                                   batchSize=batchSize, epochs=epochs)

    plt.figure()
    historyPlot(history)
    saveFig('history')

    autoencoderModel.save()
    autoencoderModel.save_weights()
    plt.show()
