import os.path

import h5py
import numpy as np
from matplotlib import pyplot as plt

from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Lambda, Reshape, Conv1D, MaxPooling1D, AveragePooling1D, UpSampling1D, Conv1DTranspose
from tensorflow.keras.models import Model
import tensorflow as tf

jetList = np.array([])

datafiles = ['~/Documents/CMEPDA/CMEPDA_exam/Data/jetImage_7_100p_30000_40000.h5']
for fileIN in datafiles:
    f = h5py.File(os.path.expanduser(fileIN))
    # for pT, etarel, phirel [5, 8, 11]
    myJetList = np.array(f.get("jetConstituentList")[:, :, [5, 8, 11]])
    jetList = np.concatenate([jetList, myJetList], axis=0) if jetList.size else myJetList
    del myJetList
    f.close()

pt_norm = 500.
#jetList[:, :, 0] = jetList[:, :, 0] / pt_norm
jetList[:, :, 1:] = jetList[:, :, 1:] * pt_norm

#jetList = jetList[:10000, :, :]

njets = jetList.shape[0]
jet_shape = jetList.shape[1:]

mse_between_inputs = tf.keras.metrics.mean_squared_error(jetList[0, :, :], jetList[1, :, :])
mse_between_inputs = np.mean(mse_between_inputs)
print(f"initial mse = {mse_between_inputs}")

def sample_latent_features(distribution):
    mean, log_variance = distribution
    random = tf.random.normal(shape=tf.shape(log_variance))
    return mean + tf.exp(1/2 * log_variance) * random

def kl_divergence_normal(distribution):
    mean, log_variance = distribution
    return 1/2 * (tf.exp(log_variance) + tf.square(mean) -1 - log_variance)

enc_dimensions = 2

encoder_input = Input(shape=jet_shape)
hidden = Conv1D(128, 9, activation="relu")(encoder_input)
hidden = MaxPooling1D(2)(hidden)
hidden = Conv1D(128, 7, activation="relu")(hidden)
hidden = AveragePooling1D(2)(hidden)
hidden = Conv1D(128, 5, activation="relu")(hidden)
hidden = MaxPooling1D(2)(hidden)
hidden = Conv1D(128, 3, activation="relu")(hidden)
hidden = AveragePooling1D(2)(hidden)
hidden = Conv1D(128, 1, activation="relu")(hidden)
hidden = AveragePooling1D(2)(hidden)
hidden = Flatten()(hidden)
hidden = Dense(64, activation="relu")(hidden)
hidden = Dense(32, activation="relu")(hidden)
hidden = Dense(16, activation="relu")(hidden)
hidden = Dense(8, activation="relu")(hidden)

mean_layer = Dense(enc_dimensions, activation="relu", name='mean')(hidden)
log_variance_layer = Dense(enc_dimensions, activation="relu", name='log_variance')(hidden)

latent_encoding = Lambda(sample_latent_features,
                         name='latent_encoding')([mean_layer, log_variance_layer])
kl_divergence = Lambda(kl_divergence_normal,
                       name='kl_divergence')([mean_layer, log_variance_layer])

decoder_input = latent_encoding
hidden = Dense(8, activation="relu")(decoder_input)
hidden = Dense(16, activation="relu")(hidden)
hidden = Dense(32, activation="relu")(hidden)
hidden = Dense(64, activation="relu")(hidden)
hidden = Dense(32, activation="relu")(hidden)
hidden = Reshape(target_shape=(1,32))(hidden)
hidden = UpSampling1D(3)(hidden)
hidden = Conv1DTranspose(128, 1, activation="relu")(hidden)
hidden = UpSampling1D(2)(hidden)
hidden = Conv1DTranspose(128, 3, activation="relu")(hidden)
hidden = UpSampling1D(2)(hidden)
hidden = Conv1DTranspose(128, 5, activation="relu")(hidden)
hidden = UpSampling1D(2)(hidden)
hidden = Conv1DTranspose(128, 7, activation="relu")(hidden)
hidden = UpSampling1D(2)(hidden)
decoder_output = Conv1DTranspose(3, 9, activation="relu", name='decoder_output')(hidden)

autoencoder_model = Model(inputs=encoder_input, outputs=[decoder_output, kl_divergence],
                    name='autoencoder')
autoencoder_model.summary()

encoder_model = Model(inputs=encoder_input, outputs=latent_encoding,
                      name='encoder')
encoder_model.compile(loss='mse', optimizer='adam')
# encoder_model.summary()

decoder_model = Model(inputs=decoder_input, outputs=decoder_output,
                      name='decoder')
decoder_model.compile(loss='mse', optimizer='adam')
# decoder_model.summary()

losses = {'decoder_output': 'mse', 'kl_divergence': 'mean_absolute_error'}
loss_weights = {'decoder_output': 1.0, 'kl_divergence': 0.01}

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

autoencoder_model.compile(loss=losses, loss_weights=loss_weights, optimizer=optimizer)

target_kl = np.zeros((njets, 1))

validation_split = 0.5
batch_size = 100
epochs = 30

history = autoencoder_model.fit(
    jetList, {'decoder_output': jetList, 'kl_divergence': target_kl},
    validation_split=validation_split, batch_size=batch_size, epochs=epochs, verbose=2,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1000, verbose=1),
               tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1000, verbose=1)])

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