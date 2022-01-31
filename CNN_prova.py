import os.path

import h5py
import numpy as np
from matplotlib import pyplot as plt

from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Lambda, Reshape, Conv1D, MaxPooling1D
from tensorflow.keras.models import Model
import tensorflow as tf

jetList = np.array([])
target = np.array([])

datafiles = ['~/Documents/CMEPDA/CMEPDA_exam/Data/jetImage_7_100p_30000_40000.h5']
for fileIN in datafiles:
    f = h5py.File(os.path.expanduser(fileIN))
    # for pT, etarel, phirel [5, 8, 11]
    myJetList = np.array(f.get("jetConstituentList")[:, :, [5, 8, 11]])
    mytarget = np.array(f.get('jets')[:, -6:-1])
    jetList = np.concatenate([jetList, myJetList], axis=0) if jetList.size else myJetList
    target = np.concatenate([target, mytarget], axis=0) if target.size else mytarget
    del myJetList, mytarget
    f.close()

pt_norm = 500.
jetList[:, :, 0] = jetList[:, :, 0] / pt_norm

njets = jetList.shape[0]
jet_shape = jetList.shape[1:]
target_shape = target.shape[1:]

print(jet_shape)

encoder_input = Input(shape=jet_shape)
hidden = Conv1D(90, 5, activation="relu")(encoder_input)
hidden = MaxPooling1D(2)(hidden)
hidden = Dropout(0.1)(hidden)
hidden = Conv1D(40, 5, activation="relu")(hidden)
hidden = MaxPooling1D(2)(hidden)
hidden = Dropout(0.1)(hidden)
hidden = Conv1D(15, 5, activation="relu")(hidden)
hidden = MaxPooling1D(2)(hidden)
hidden = Flatten()(hidden)
hidden = Dense(10, activation="relu")(hidden)
hidden = Dense(10, activation="relu")(hidden)
hidden = Dense(10, activation="relu")(hidden)
encoder_output = Dense(5, activation="sigmoid")(hidden)

encoder_model = Model(inputs=encoder_input, outputs=encoder_output,
                      name='encoder')
encoder_model.summary()

losses = 'binary_crossentropy'
loss_weights = 1.0

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

encoder_model.compile(loss=losses, loss_weights=loss_weights, optimizer=optimizer)

history = encoder_model.fit(
    jetList, target,
    validation_split=0.8, batch_size=10, epochs=1000, verbose=2,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1),
               tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)])

print(history.history.keys())
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.yscale('log')
plt.grid()

encoder_model.save('Trained_Models/cnn')

plt.show()