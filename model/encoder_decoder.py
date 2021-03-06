"""Construct encoder and decoder models' structure to put into class vae"""

import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Lambda, Concatenate)


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


def enc_dec_layer_builder(jet_shape=5, enc_dimensions=3, target_shape=5,
                          enc_hidden_nodes=(64, 64, 32, 16, 8),
                          dec_hidden_nodes=(8, 16, 32, 64, 64)):
    """
    Construct the autoencoder layers and connections.

        Parameters:
            jet_shape : int
                Number of jet features in input
                Default: 5
            enc_dimensions : int
                Dimension of the encoding space
                Default: 3
            target_shape : int
                number of jet types
                Default: 5
            enc_hidden_nodes : list or tuple
                list of hidden dense layers' shapes for the encoder
                Default: (64, 64, 32, 16, 8)
            dec_hidden_nodes : list or tuple
                list of hidden dense layers' shapes for the decoder
                Default: (8, 16, 32, 64, 64)

        Returns:
            [vae_input, vae_output, encoder_input, encoder_output,
             decoder_input, decoder_output]
                a list containing input and output layers for the vae,
                encoder and decoder models
    """

    jet_input = Input(shape=jet_shape, name='jet_input')
    target_input = Input(shape=target_shape, name='target_input')

    encoder_input = Concatenate()([jet_input, target_input])

    hidden = encoder_input
    for u in enc_hidden_nodes:
        hidden = Dense(u, activation="relu")(hidden)

    mean_layer = Dense(enc_dimensions, activation="linear",
                       name='mean')(hidden)
    log_variance_layer = Dense(
        enc_dimensions, activation="linear", name='log_variance')(hidden)

    latent_encoding = Lambda(sample_latent_features,
                             name='latent_encoding')([mean_layer,
                                                      log_variance_layer])
    encoder_output = latent_encoding
    kl_divergence = Lambda(kl_divergence_normal,
                           name='kl_divergence')([mean_layer, log_variance_layer])

    decoder_input = Concatenate()([latent_encoding, target_input])

    hidden = decoder_input
    for u in dec_hidden_nodes:
        hidden = Dense(u, activation="relu")(hidden)

    decoder_output = Dense(
        jet_shape, activation="linear", name='decoder_output')(hidden)

    vae_input = [jet_input, target_input]
    vae_output = [decoder_output, kl_divergence]

    return [vae_input, vae_output, encoder_input, encoder_output,
            decoder_input, decoder_output]
