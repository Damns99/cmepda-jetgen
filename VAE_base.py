import h5py
import numpy as np
import matplotlib.pyplot as plt
import os.path

from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Lambda, Reshape
from tensorflow.keras.models import Model
import tensorflow as tf

## questa riga fa funzionare, quali sono però i lati negativi?
# la eager execution è una cosa default di TF 2, mentre in TF 1 era disattivata
tf.compat.v1.disable_eager_execution()

#target = np.array([])
jetList = np.array([])
# we cannot load all data on Colab. So we just take a few files
datafiles = ['~/Documents/CMEPDA/CMEPDA_exam/Data/jetImage_7_100p_30000_40000.h5']
# if you are running locallt, you can use the full dataset doing
# for fileIN in glob.glob("tutorials/HiggsSchool/data/*h5"):
for fileIN in datafiles:
    f = h5py.File(os.path.expanduser(fileIN))
    # for pT, etarel, phirel
    myJetList = np.array(f.get("jetConstituentList")[:,:,[5,8,11]])
    # mytarget = np.array(f.get('jets')[0:,-6:-1])
    jetList = np.concatenate([jetList, myJetList], axis=0) if jetList.size else myJetList
    #target = np.concatenate([target, mytarget], axis=0) if target.size else mytarget
    del myJetList#, mytarget
    f.close()
print(jetList.shape[1:])

def sample_latent_features(distribution):
    distribution_mean, distribution_variance = distribution
    batch_size = tf.shape(distribution_variance)[0]
    random = tf.keras.backend.random_normal(shape=(batch_size, tf.shape(distribution_variance)[1]))
    return distribution_mean + tf.exp(-0.5 * distribution_variance) * random

# encoding_dimension_number = 2

encoder_input = Input(shape = jetList.shape[1:])
hidden = Flatten()(encoder_input)
hidden = Dense(16, activation="relu")(hidden)
hidden = Dense(16, activation="relu")(hidden)
hidden = Dense(16, activation="relu")(hidden)

distribution_mean = Dense(2, name='mean')(hidden)
distribution_variance = Dense(2, name='log_variance')(hidden)
latent_encoding = Lambda(sample_latent_features)([distribution_mean, distribution_variance])

encoder_model = Model(encoder_input, latent_encoding)
# encoder_model.summary()

decoder_input = Input(shape = (2))
hiddenn = Dense(16, activation="relu")(decoder_input)
hiddenn = Dense(16, activation="relu")(hiddenn)
hiddenn = Dense(16, activation="relu")(hiddenn)
hiddenn = Dense(100*3, activation="relu")(hiddenn)
decoder_output = Reshape(target_shape = (100, 3))(hiddenn)

decoder_model = Model(decoder_input, decoder_output)
# decoder_model.summary()

encoded = encoder_model(encoder_input)
decoded = decoder_model(encoded)
autoencoder = Model(encoder_input, decoded)
autoencoder.summary(expand_nested=True)

def get_loss(distribution_mean, distribution_variance):

    def get_reconstruction_loss(y_true, y_pred):
        reconstruction_loss = tf.keras.losses.mse(y_true, y_pred)
        reconstruction_loss_batch = tf.reduce_mean(reconstruction_loss)
        return reconstruction_loss_batch*100*3

    def get_kl_loss(distribution_mean, distribution_variance):
        kl_loss = 1/2 * (distribution_variance + tf.square(distribution_mean) -1 - tf.math.log(distribution_variance))
        kl_loss_batch = tf.reduce_mean(kl_loss)
        return kl_loss_batch

    def total_loss(y_true, y_pred):
        reconstruction_loss_batch = get_reconstruction_loss(y_true, y_pred)
        kl_loss_batch = get_kl_loss(distribution_mean, distribution_variance)
        return reconstruction_loss_batch + kl_loss_batch

    return total_loss

autoencoder.compile(loss = get_loss(distribution_mean, distribution_variance), optimizer = 'adam')
# autoencoder.compile(loss = 'mse', optimizer = 'adam')

history = autoencoder.fit(jetList, jetList, validation_split = 0.5, epochs = 5, verbose = 1)
