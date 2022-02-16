"""Define model class vae using layers defined in model.encoder_decoder"""

import os
import numpy as np
import tensorflow as tf
from model.encoder_decoder import (
    encoder_model, encoder_input, decoder_model, decoder_output,
    kl_divergence, enc_dimensions, jet_input, target_input, classification)

train_path = os.path.join(os.path.dirname(__file__), '..', 'trained_models')


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

    def __init__(self):
        """
        Construct attributes for the vae model from layers and models defined
        in model.encoder_decoder.
        """

        super(vae, self).__init__(inputs=encoder_input,
                                  outputs=[decoder_output,
                                           kl_divergence, classification],
                                  name='autoencoder')
        self.encoder = encoder_model
        self.decoder = decoder_model
        self.enc_dimensions = enc_dimensions
        self.my_losses = {'decoder_output': 'mse',
                          'kl_divergence': 'mean_absolute_error',
                          'classification': 'binary_crossentropy'}

    def compile(self, learning_rate=0.001, loss_weights=(1.0, 1.0, 1.0), **kwargs):
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
            'kl_divergence': loss_weights[1],
            'classification': loss_weights[2]}
        super(vae, self).compile(loss=self.my_losses,
                                 optimizer=self.my_optimizer,
                                 loss_weights=self.my_loss_weights, **kwargs)

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
        return super(vae, self).fit(jet_list,
                                    {'decoder_output': jet_list,
                                     'kl_divergence': target_kl,
                                     'classification': target},
                                    batch_size=batch_size,
                                    validation_split=validation_split,
                                    epochs=epochs, verbose=2, **kwargs)

    def encoder_predict(self, jet_list, **kwargs):
        """
        Generate encoded features and jet type predictions for the input jet features

        Parameters:
            jet_list: numpy 2d array
                Input array of rows of model.encoder_decoder.jetShape jet features
        Returns:
            Numpy array(s) of predictions.
        """

        return self.encoder.predict(
            tf.concat(jet_list, axis=-1), **kwargs)

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

    def load_from_file(self, custom_name=''):
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
