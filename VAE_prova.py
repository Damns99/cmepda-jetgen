"""Open the dataset, create a vae model and train it, saving history and weights."""

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

from utilities.file_opener import getJetList, standData
from utilities.plots import historyPlot
from utilities.figure_saver import saveFig
from model.vae import vae

with tf.device('/GPU:0'):
    jetList, target = getJetList()

    jetList = standData(jetList)

    initialLoss = [tf.keras.metrics.mse(jetList[0, :], jetList[i, :]) for i in range(1,101)]
    print(f'initialLoss = {np.mean(initialLoss)}')

    autoencoderModel = vae()
    autoencoderModel.summary()

    lossWeights = [1.0, 0.05]
    learningRate = 0.0001

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
