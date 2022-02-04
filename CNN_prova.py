import os.path

import h5py
import numpy as np
from matplotlib import pyplot as plt

from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Lambda, Reshape, Conv1D, MaxPooling1D
from tensorflow.keras.models import Model
import tensorflow as tf

from utilities.file_opener import getJetList

jetList, target = getJetList()

pt_norm = 500.
jetList[:, :, 0] = jetList[:, :, 0] / pt_norm

#jetList = jetList[:10000, :, :]
#target = target[:10000, :]

njets = jetList.shape[0]
jet_shape = jetList.shape[1:]
target_shape = target.shape[1:]

encoder_input = Input(shape=jet_shape)
hidden = Conv1D(64, 7, activation="relu")(encoder_input)
hidden = MaxPooling1D(2)(hidden)
hidden = Conv1D(64, 5, activation="relu")(hidden)
hidden = MaxPooling1D(2)(hidden)
hidden = Conv1D(32, 3, activation="relu")(hidden)
hidden = MaxPooling1D(2)(hidden)
hidden = Conv1D(32, 3, activation="relu")(hidden)
hidden = MaxPooling1D(2)(hidden)
hidden = Flatten()(hidden)
hidden = Dropout(0.4)(hidden)
hidden = Dense(32, activation="relu")(hidden)
hidden = Dropout(0.2)(hidden)
hidden = Dense(32, activation="relu")(hidden)
hidden = Dense(32, activation="relu")(hidden)
hidden = Dense(16, activation="relu")(hidden)
hidden = Dropout(0.2)(hidden)
hidden = Dense(16, activation="relu")(hidden)
hidden = Dense(16, activation="relu")(hidden)
hidden = Dense(16, activation="relu")(hidden)
hidden = Dropout(0.2)(hidden)
hidden = Dense(8, activation="relu")(hidden)
encoder_output = Dense(5, activation="sigmoid")(hidden)

encoder_model = Model(inputs=encoder_input, outputs=encoder_output,
                      name='encoder')
encoder_model.summary()

losses = 'categorical_crossentropy'
loss_weights = 1.0

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

encoder_model.compile(loss=losses, loss_weights=loss_weights, optimizer=optimizer)

validation_split = 0.5
batch_size = 100
epochs = 500

history = encoder_model.fit(
    jetList, target,
    validation_split=validation_split, batch_size=batch_size, epochs=epochs, verbose=2,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=200, verbose=1),
               tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=25, verbose=1)])

print(f'history keys: {history.history.keys()}')
plt.plot(history.history["loss"])
if validation_split > 0.:
    plt.plot(history.history["val_loss"])
plt.yscale('log')
plt.grid()

encoder_model.save('Trained_Models/cnn')

plt.show()