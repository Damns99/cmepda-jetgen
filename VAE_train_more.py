import os.path

import h5py
import numpy as np
from matplotlib import pyplot as plt

from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Lambda, Reshape
from tensorflow.keras.models import Model
import tensorflow as tf

from utilities.file_opener import getJetList

jetList, _ = getJetList()

#pt_norm = 500.
#jetList[:, :, 0] = jetList[:, :, 0] / pt_norm

#jetList = jetList[:10000, :, :]

njets = jetList.shape[0]
jet_shape = jetList.shape[1:]

encoder_model = tf.keras.models.load_model('Trained_Models/encoder')
decoder_model = tf.keras.models.load_model('Trained_Models/decoder')
autoencoder_model = tf.keras.models.load_model('Trained_Models/autoencoder')

# Change learning rate if you want
new_learning_rate = 0.001
tf.keras.backend.set_value(autoencoder_model.optimizer.learning_rate,
                           new_learning_rate)

target_kl = np.zeros((njets, 1))

validation_split = 0.5
batch_size = 800
epochs = 5

history = autoencoder_model.fit(
    jetList, {'decoder_output': jetList, 'kl_divergence': target_kl},
    validation_split=validation_split, batch_size=batch_size, epochs=epochs, verbose=2,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, verbose=1),
               tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=25, verbose=1)])

print(history.history.keys())
plt.plot(history.history["loss"])
if validation_split > 0.:
    plt.plot(history.history["val_loss"])
plt.yscale('log')
plt.grid()

autoencoder_model.save('Trained_Models/autoencoder')
encoder_model.save('Trained_Models/encoder')
decoder_model.save('Trained_Models/decoder')

plt.show()