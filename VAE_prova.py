import h5py
import numpy as np
import matplotlib.pyplot as plt
import os.path

from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Lambda, Reshape
from tensorflow.keras.models import Model
import tensorflow as tf

jetList = np.array([])

datafiles = ['~/Documents/CMEPDA/CMEPDA_exam/Data/jetImage_7_100p_30000_40000.h5']
for fileIN in datafiles:
    f = h5py.File(os.path.expanduser(fileIN))
    # for pT, etarel, phirel
    myJetList = np.array(f.get("jetConstituentList")[:,:,[5,8,11]])
    jetList = np.concatenate([jetList, myJetList], axis=0) if jetList.size else myJetList
    del myJetList
    f.close()

def sample_latent_features(distribution):
    distribution_mean, distribution_variance = distribution
    distribution_variance = tf.square(distribution_variance)
    batch_size = tf.shape(distribution_variance)[0]
    random = tf.keras.backend.random_normal(shape=(batch_size, tf.shape(distribution_variance)[1]))
    return distribution_mean + tf.exp(-0.5 * distribution_variance) * random

def kl_divergence_normal(distribution):
    distribution_mean, distribution_variance = distribution
    distribution_variance = tf.square(distribution_variance)
    return 1/2 * (distribution_variance + tf.square(distribution_mean) -1 - tf.math.log(distribution_variance))

enc_dimensions = 2

encoder_input = Input(shape = jetList.shape[1:])
hidden = Flatten()(encoder_input)
hidden = Dense(64, activation="relu")(hidden)
hidden = Dense(32, activation="relu")(hidden)
hidden = Dense(16, activation="relu")(hidden)
hidden = Dense(8, activation="relu")(hidden)

distribution_mean = Dense(2, name = 'mean')(hidden)
distribution_variance = Dense(2, name = 'log_variance')(hidden)
latent_encoding = Lambda(sample_latent_features, name = 'latent_encoding')([distribution_mean, distribution_variance])
kl_divergence = Lambda(kl_divergence_normal, name = 'kl_divergence')([distribution_mean, distribution_variance])

decoder_input = latent_encoding
hidden = Dense(8, activation="relu")(decoder_input)
hidden = Dense(16, activation="relu")(hidden)
hidden = Dense(32, activation="relu")(hidden)
hidden = Dense(64, activation="relu")(hidden)
hidden = Dense(100*3, activation="relu")(hidden)
decoder_output = Reshape(target_shape = (100, 3), name = 'decoder_output')(hidden)

autoencoder = Model(inputs = encoder_input, outputs = [decoder_output, kl_divergence], name = 'autoencoder')
autoencoder.summary()

encoder_model = Model(inputs = encoder_input, outputs = latent_encoding, name = 'encoder')
# encoder_model.summary()

decoder_model = Model(inputs = decoder_input, outputs = decoder_output, name = 'decoder')
# decoder_model.summary()

losses = {'decoder_output': 'mse', 'kl_divergence': 'mean_absolute_error'}
loss_weights = {'decoder_output': 1.0, 'kl_divergence': 1.0}

autoencoder.compile(loss = losses, loss_weights = loss_weights, optimizer = 'adam')

target_normal = np.zeros((jetList.shape[0], 1))

history = autoencoder.fit(jetList, {'decoder_output': jetList, 'kl_divergence': target_normal}, validation_split = 0.5, batch_size = 50, epochs = 100, verbose = 1)
