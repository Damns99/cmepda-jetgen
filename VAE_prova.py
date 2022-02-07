import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

from utilities.file_opener import getJetList
from utilities.plots import historyPlot
from model.vae import vae

with tf.device('/CPU:0'):
    jetList, _ = getJetList()

    mseBetweenInputs = tf.keras.metrics.mean_squared_error(
        jetList[0, :, :], jetList[1, :, :])
    mseBetweenInputs = np.mean(mseBetweenInputs)
    print(f"initial mse = {mseBetweenInputs}")

    autoencoderModel = vae()
    autoencoderModel.summary()

    lossWeights = [1.0, 0.1]
    learningRate = 0.001

    autoencoderModel.compile(lossWeights=lossWeights,
                             learningRate=learningRate)

    validationSplit = 0.5
    batchSize = 800
    epochs = 1

    history = autoencoderModel.fit(jetList, validationSplit=validationSplit,
                                   batchSize=batchSize, epochs=epochs)

    historyPlot(history)

    autoencoderModel.save()
    autoencoderModel.save_weights()
    plt.show()
