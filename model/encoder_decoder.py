"""Construct encoder and decoder models' structure to put into class vae"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Lambda, Concatenate)

jetShape = 5
encDimensions = 3
targetShape = 5


def sample_latent_features(distribution):
    """
    Return random gaussian points in the encoding space,
    given means and variances on each encoding axis.

    Parameters:
        distribution : list
            A list of two tensors with same shape, each row representing
            means and variances on each encoding axis for each jet.

    Returns:
        A tensor with the same shape of the two inputs filled with random
        gaussian numbers chosen using corresponding mean and variance.
    """
    mean, log_variance = distribution
    random = tf.random.normal(shape=tf.shape(log_variance))
    return mean + tf.exp(1/2 * log_variance) * random


def kl_divergence_normal(distribution):
    """
    Compute the Kullback–Leibler divergences of a series of gaussian
    distributions with respect to a normal distribution, to evaluate
    their difference.

        Parameters:
            distribution : list
                A list of two tensors with same shape, each row representing
                means and variances on each encoding axis for each jet.

        Returns:
            A tensor with the same shape of the two inputs filled with the
            computed KL divergences.
    """
    mean, log_variance = distribution
    return 1/2 * (tf.exp(log_variance) + tf.square(mean) - 1 - log_variance)


jet_input = Input(shape=jetShape, name='jet_input')
target_input = Input(shape=targetShape, name='target_input')

encoder_input = Concatenate()([jet_input, target_input])

hidden = Dense(64, activation="relu")(encoder_input)
hidden = Dense(64, activation="relu")(hidden)
hidden = Dense(32, activation="relu")(hidden)
hidden = Dense(16, activation="relu")(hidden)
hidden = Dense(8, activation="relu")(hidden)

mean_layer = Dense(encDimensions, activation="linear", name='mean')(hidden)
log_variance_layer = Dense(
    encDimensions, activation="linear", name='log_variance')(hidden)

latent_encoding = Lambda(sample_latent_features,
                         name='latent_encoding')([mean_layer,
                                                  log_variance_layer])
kl_divergence = Lambda(kl_divergence_normal,
                       name='kl_divergence')([mean_layer, log_variance_layer])

decoder_input = Concatenate()([latent_encoding, target_input])

hidden = Dense(8, activation="relu")(decoder_input)
hidden = Dense(16, activation="relu")(hidden)
hidden = Dense(32, activation="relu")(hidden)
hidden = Dense(64, activation="relu")(hidden)
hidden = Dense(64, activation="relu")(hidden)
decoder_output = Dense(5, activation='linear', name='decoder_output')(hidden)


encoder_model = Model(inputs=encoder_input, outputs=latent_encoding,
                      name='encoder')
encoder_model.compile(loss='mse', optimizer='adam')

decoder_model = Model(inputs=decoder_input, outputs=decoder_output,
                      name='decoder')
decoder_model.compile(loss='mse', optimizer='adam')
