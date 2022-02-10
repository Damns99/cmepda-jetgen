import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

from sklearn.cluster import Birch

from utilities.file_opener import getJetList
from utilities.model_getter import getModels
from utilities.plots import historyPlot, jetScatter, jetScatter3D
from utilities.figure_saver import saveFig
from model.vae import vae

w1 = 1000
w3 = 100


with tf.device('/CPU:0'):
    jetList, target = getJetList()

    #jetList = jetList[:1000, :]
    #target = target[:1000, :]

    njets = jetList.shape[0]

    autoencoder_model = getModels()

    jetTag = np.argmax(target, axis=1)

    jetList[:, 0] = jetList[:, 0] / w1
    jetList[:, 2] = jetList[:, 2] / w3
    jetList[:, 1] = np.abs(jetList[:, 1])

    encoded_out = autoencoder_model.encoder_predict(jetList=jetList)
    encoded_features, pred_target = encoded_out
    decoded_jets = autoencoder_model.decoder_predict(tf.concat(encoded_out, axis=-1))

    print(decoded_jets[:10, :])
    print(jetList[:10, :])

    plt.figure()
    plt.subplot(121)
    jetScatter(encoded_features, jetTag, 0, 1)
    plt.subplot(122)
    jetScatter(encoded_features, jetTag, 0, 2)


    plt.figure()
    jetScatter3D(encoded_features, jetTag)

    #brc = Birch(branching_factor=50, n_clusters=5, threshold=0.1)
    #brc.fit(encoded_features)
    #predTag = brc.predict(encoded_features)

    #plt.figure()
    #jetScatter(encoded_features, predTag)

    print(pred_target[0:20])

    part_pred = np.argmax(pred_target, axis=1)
    part_real = np.argmax(target, axis=1)

    print(part_pred[0:20])
    print(part_real[0:20])

    correct = (part_pred == part_real)
    prediction_accuracy = np.count_nonzero(correct) / njets
    print(f'pred acc. = {prediction_accuracy * 100 : .2f} %')

    njets_per_type = np.count_nonzero(target, axis=0)
    pred_acc_per_type = np.array([
            np.count_nonzero(np.logical_and(correct, part_real == i)) / n
            for i, n in enumerate(njets_per_type)])
    for i, pacc in enumerate(pred_acc_per_type):
        print(f'particle {i} pred acc. = {pacc * 100 : .2f} %')

    plt.show()
