import os.path

import h5py
import numpy as np

from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Lambda, Reshape
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

pt_norm = 100.
jetList[:, :, 0] = jetList[:, :, 0] / pt_norm

njets = jetList.shape[0]
jet_shape = jetList.shape[1:]

def sample_latent_features(distribution):
    mean, log_variance = distribution
    random = tf.random.normal(shape=tf.shape(log_variance))
    return mean + tf.exp(1/2 * log_variance) * random

def kl_divergence_normal(distribution):
    mean, log_variance = distribution
    return 1/2 * (tf.exp(log_variance) + tf.square(mean) -1 - log_variance)

enc_dimensions = 2

encoder_input = Input(shape=jet_shape)
hidden = Flatten()(encoder_input)
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
hidden = Dense(np.prod(jet_shape), activation="relu")(hidden)
decoder_output = Reshape(target_shape=jet_shape, name='decoder_output')(hidden)

autoencoder = Model(inputs=encoder_input, outputs=[decoder_output, kl_divergence],
                    name='autoencoder')
autoencoder.summary()

encoder_model = Model(inputs=encoder_input, outputs=latent_encoding,
                      name='encoder')
# encoder_model.summary()

decoder_model = Model(inputs=decoder_input, outputs=decoder_output,
                      name='decoder')
# decoder_model.summary()

losses = {'decoder_output': 'mse', 'kl_divergence': 'mean_absolute_error'}
loss_weights = {'decoder_output': 1.0, 'kl_divergence': 0.1}

optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

autoencoder.compile(loss=losses, loss_weights=loss_weights, optimizer=optimizer)

target_kl = np.zeros((njets, 1))

history = autoencoder.fit(
    jetList, {'decoder_output': jetList, 'kl_divergence': target_kl},
    validation_split=0.5, batch_size=10, epochs=10, verbose=1)
