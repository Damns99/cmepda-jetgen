"""Open the dataset, create a vae model and train it, saving history and weights."""

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

from utilities.file_opener import get_jet_list, stand_data
from utilities.plots import history_plot, save_figure
from model.vae import vae

with tf.device('/GPU:0'):
    jet_list, target = get_jet_list()

    jet_list = stand_data(jet_list)

    initial_loss = [tf.keras.metrics.mse(jet_list[0, :], jet_list[i, :])
                    for i in range(1, 101)]
    print(f'initial_loss = {np.mean(initial_loss)}')

    autoencoder_model = vae()
    autoencoder_model.summary()

    loss_weights_list = [1.0, 0.05]
    learning_rate = 0.0001

    autoencoder_model.compile(loss_weights_list=loss_weights_list,
                              learning_rate=learning_rate)

    validation_split = 0.5
    batch_size = 400
    epochs = 300

    history = autoencoder_model.fit(jet_list, target,
                                    validation_split=validation_split,
                                    batch_size=batch_size, epochs=epochs)

    plt.figure()
    history_plot(history)
    save_figure('history')

    autoencoder_model.save()
    autoencoder_model.save_weights()
    plt.show()
