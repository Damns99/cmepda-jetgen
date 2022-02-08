import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

from utilities.file_opener import getJetList
from utilities.plots import historyPlot
from model.vae import vae

w1 = 1000
w3 = 100

with tf.device('/CPU:0'):
    jetList, _ = getJetList()

    jetList[:, 0] = jetList[:, 0] / w1
    jetList[:, 2] = jetList[:, 2] / w3
    jetList[:, 1] = np.abs(jetList[:, 1])

    autoencoderModel = vae()
    autoencoderModel.summary()

    lossWeights = [1.0, 0.1]
    learningRate = 0.001

    autoencoderModel.compile(lossWeights=lossWeights,
                             learningRate=learningRate)

    validationSplit = 0.5
    batchSize = 800
    epochs = 500

    history = autoencoderModel.fit(jetList, validationSplit=validationSplit,
                                   batchSize=batchSize, epochs=epochs)

    historyPlot(history)

    autoencoderModel.save()
    autoencoderModel.save_weights()
    plt.show()
