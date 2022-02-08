import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Lambda)

jetShape = (5)
encDimensions = 2


def sample_latent_features(distribution):
    mean, log_variance = distribution
    random = tf.random.normal(shape=tf.shape(log_variance))
    return mean + tf.exp(1/2 * log_variance) * random


def kl_divergence_normal(distribution):
    mean, log_variance = distribution
    return 1/2 * (tf.exp(log_variance) + tf.square(mean) - 1 - log_variance)


encoder_input = Input(shape=jetShape)

hidden = Dense(64, activation="relu")(encoder_input)
hidden = Dense(32, activation="relu")(hidden)
hidden = Dense(16, activation="relu")(hidden)
hidden = Dense(8, activation="relu")(hidden)

mean_layer = Dense(encDimensions, activation="relu", name='mean')(hidden)
log_variance_layer = Dense(
    encDimensions, activation="relu", name='log_variance')(hidden)

latent_encoding = Lambda(sample_latent_features,
                         name='latent_encoding')([mean_layer,
                                                 log_variance_layer])
kl_divergence = Lambda(kl_divergence_normal,
                       name='kl_divergence')([mean_layer, log_variance_layer])

decoder_input = latent_encoding

hidden = Dense(8, activation="relu")(decoder_input)
hidden = Dense(16, activation="relu")(hidden)
hidden = Dense(32, activation="relu")(hidden)
hidden = Dense(64, activation="relu")(hidden)
hidden = Dense(32, activation="relu")(hidden)
decoder_output = Dense(5, activation='relu', name='decoder_output')(hidden)


encoder_model = Model(inputs=encoder_input, outputs=latent_encoding,
                      name='encoder')
encoder_model.compile(loss='mse', optimizer='adam')

decoder_model = Model(inputs=decoder_input, outputs=decoder_output,
                      name='decoder')
decoder_model.compile(loss='mse', optimizer='adam')
