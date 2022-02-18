"""Define model class vae using layers defined in model.encoder_decoder"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from model.encoder_decoder import enc_dec_layer_builder

train_path = os.path.join(os.path.dirname(__file__), '..', 'trained_models')


@tf.keras.utils.register_keras_serializable()
class vae(tf.keras.Model):
    """
    A class representing a Variation Auto-Encoder, extending tf.keras.Model

    Attributes:
        encoder : tf.keras.Model
            The encoder sub-model as defined and compiled in model.encoder_decoder
        decoder : tf.keras.Model
            The decoder sub-model as defined and compiled in model.encoder_decoder
        enc_dimensions : int
            Dimension of the encoding space
        my_losses : dict
            Losses to evaluate for the corresponding output layer of the vae

    Methods:
        compile(self, learning_rate=0.001, loss_weights=[1.0, 1.0, 1.0], **kwargs):
            Compiles the model
        fit(self, jet_list, target, validation_split=0.5, batch_size=800, epochs=30,
            **kwargs):
            Train the model
        encoder_predict(self, jet_list, **kwargs):
            Predict encoder output on given input data
        decoder_predict(self, encoded_features, **kwargs):
            Predict decoder output on given input data
        save(self, custom_name=''):
            Save vae, encoder and decoder models for later use
        save_weights(self, custom_name=''):
            Save vae, encoder and decoder models' weights in .h5 files
        load_from_file(self, custom_name=''):
            Load vae, encoder and decoder models' weights from .h5 files
    """

    def __init__(self, jet_shape=5, enc_dimensions=3, target_shape=5,
                 enc_hidden_nodes=(64, 64, 32, 16, 8),
                 dec_hidden_nodes=(8, 16, 32, 64, 64), **kwargs):
        """
        Construct attributes for the vae model from layers and models defined
        in model.encoder_decoder.
        """

        self.jet_shape = jet_shape
        self.enc_dimensions = enc_dimensions
        self.target_shape = target_shape
        self.enc_hidden_nodes = enc_hidden_nodes
        self.dec_hidden_nodes = dec_hidden_nodes

        built_enc_dec = enc_dec_layer_builder(self.jet_shape,
                                              self.enc_dimensions,
                                              self.target_shape,
                                              self.enc_hidden_nodes,
                                              self.dec_hidden_nodes)
        vae_input, vae_output, encoder_input, encoder_output, \
            decoder_input, decoder_output = built_enc_dec

        super(vae, self).__init__(inputs=vae_input, outputs=vae_output,
                                  name='autoencoder')
        self.encoder = Model(inputs=encoder_input, outputs=encoder_output,
                             name='encoder')
        self.decoder = Model(inputs=decoder_input, outputs=decoder_output,
                             name='decoder')
        self.myLosses = {'decoder_output': 'mse',
                         'kl_divergence': 'mean_absolute_error'}

    def get_config(self):
        config = super(vae, self).get_config()
        config["layers"].pop(-1)
        config["layers"].pop(-1)
        config.update({"jet_shape": self.jet_shape,
                       "enc_dimensions": self.enc_dimensions,
                       "target_shape": self.target_shape,
                       "enc_hidden_nodes": self.enc_hidden_nodes,
                       "dec_hidden_nodes": self.dec_hidden_nodes})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compile(self, learning_rate=0.001, loss_weights=(1.0, 1.0), **kwargs):
        """
        Extend tf.keras.Model.compile to work with vae structure and parameters.

        Parameters:
            learning_rate : float
                Learning rate to use for the model's optimizer (Adam)
                Default 0.001
            loss_weights : list or tuple
                List of 2 weights for each loss to compute the total loss,
                in order 'decoder_output', 'kl_divergence', 'classification'
                Default (1.0, 1.0)
        """

        self.my_optimizer = tf.keras.optimizers.Adam(
                                    learning_rate=learning_rate)
        self.my_loss_weights = {
            'decoder_output': loss_weights[0],
            'kl_divergence': loss_weights[1]}
        super(vae, self).compile(loss=self.my_losses,
                                 optimizer=self.my_optimizer,
                                 loss_weights=self.my_loss_weights, **kwargs)

        self.encoder.compile(loss='mse', optimizer='adam')
        self.decoder.compile(loss='mse', optimizer='adam')

    def fit(self, jet_list, target, validation_split=0.5, batch_size=800,
            epochs=30, **kwargs):
        """
        Extend tf.keras.Model.fit to work with vae structure and parameters.

        Parameters:
            jet_list : numpy 2d array
                Input array of rows of model.encoder_decoder.jetShape jet features
            validation_split : float
                Fraction of inputs to use as validation set. Between 0 and 1
                Default 0.5
            batch_size : int
                Number of samples per gradient update
                Default 800
            epochs : int
                Number of training iterations on the entire dataset
                Default 30

        Returns:
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable)
        """

        target_kl = np.zeros((jet_list.shape[0], 1))
        return super(vae, self).fit({'jet_input': jet_list,
                                     'target_input': target},
                                    {'decoder_output': jet_list,
                                     'kl_divergence': target_kl},
                                    batch_size=batch_size,
                                    validation_split=validation_split,
                                    epochs=epochs, verbose=2, **kwargs)

    def encoder_predict(self, jet_list, target, **kwargs):
        """
        Generate encoded features and jet type predictions for the input jet features

        Parameters:
            jet_list: numpy 2d array
                Input array of rows of model.encoder_decoder.jetShape jet features
        Returns:
            Numpy array(s) of predictions.
        """

        return self.encoder.predict(
            tf.concat([jet_list, target], axis=-1), **kwargs)

    def decoder_predict(self, encoded_features, target, **kwargs):
        """
        Generate decoded jet predictions for the input encodings and particle types

        Parameters:
            encoded_features: numpy 2d array
                Input array of rows of model.encoder_decoder.jetShape encodings
                each with length self.encDimensions + model.encoder_decoder.targetShape
        Returns:
            Numpy array(s) of predictions.
        """

        return self.decoder.predict(
            tf.concat([encoded_features, target], axis=-1), **kwargs)

    def save(self, custom_name=''):
        """
        Save vae, encoder and decoder models as three Tensorflow SavedModel
        in the trainPath folder.

        Parameters:
            custom_name : string
                String to append at each model's folder name
                Default ''
        """

        self.encoder.save(os.path.join(
            train_path, ''.join(['encoder', custom_name])))
        self.decoder.save(os.path.join(
            train_path, ''.join(['decoder', custom_name])))
        return super(vae, self).save(os.path.join(
            train_path, ''.join(['vae', custom_name])))

    def save_weights(self, custom_name=''):
        """
        Save vae, encoder and decoder model's weights as three .h5 files
        in the trainPath folder.

        Parameters:
            custom_name : string
                String to append at each model's file name
                Default ''
        """

        self.encoder.save_weights(os.path.join(
            train_path, ''.join(['encoder', custom_name, '.h5'])))
        self.decoder.save_weights(os.path.join(
            train_path, ''.join(['decoder', custom_name, '.h5'])))
        return super(vae, self).save_weights(os.path.join(
            train_path, ''.join(['vae', custom_name, '.h5'])))

    def load_weights(self, custom_name=''):
        """
        Load vae, encoder and decoder model's weights from three .h5 files
        in the trainPath folder.

        Parameters:
            custom_name : string
                String to append at each model's file name
                Default ''

        Returns:
            Loading weights in HDF5 format, so it returns `None`
        """

        self.encoder.load_weights(os.path.join(
                train_path, ''.join(['encoder', custom_name, '.h5'])))
        self.decoder.load_weights(os.path.join(
                train_path, ''.join(['decoder', custom_name, '.h5'])))
        return super(vae, self).load_weights(os.path.join(
            train_path, ''.join(['vae', custom_name, '.h5'])))
