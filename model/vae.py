"""Define model class vae using layers defined in model.encoder_decoder"""

import os
import numpy as np
import tensorflow as tf
from model.encoder_decoder import (
    encoder_model, encoder_input, decoder_model, decoder_output,
    kl_divergence, classification, encDimensions)

trainPath = os.path.join(os.path.dirname(__file__), '..', 'trained_models')


class vae(tf.keras.Model):
    """
    A class representing a Variation Auto-Encoder, extending tf.keras.Model

    Attributes:
        encoder : tf.keras.Model
            The encoder sub-model as defined and compiled in model.encoder_decoder
        decoder : tf.keras.Model
            The decoder sub-model as defined and compiled in model.encoder_decoder
        encDimensions : int
            Dimension of the encoding space
        myLosses : dict
            Losses to evaluate for the corresponding output layer of the vae

    Methods:
        compile(self, learningRate=0.001, lossWeights=[1.0, 1.0, 1.0], **kwargs):
            Compiles the model
        fit(self, jetList, target, validationSplit=0.5, batchSize=800, epochs=30,
            **kwargs):
            Train the model
        encoder_predict(self, jetList, **kwargs):
            Predict encoder output on given input data
        decoder_predict(self, encodedFeatures, **kwargs):
            Predict decoder output on given input data
        save(self, customName=''):
            Save vae, encoder and decoder models for later use
        save_weights(self, customName=''):
            Save vae, encoder and decoder models' weights in .h5 files
        load_from_file(self, customName=''):
            Load vae, encoder and decoder models' weights from .h5 files
    """

    def __init__(self):
        """
        Construct attributes for the vae model from layers and models defined
        in model.encoder_decoder.
        """

        super(vae, self).__init__(inputs=encoder_input,
                                  outputs=[decoder_output, kl_divergence, classification],
                                  name='autoencoder')
        self.encoder = encoder_model
        self.decoder = decoder_model
        self.encDimensions = encDimensions
        self.myLosses = {'decoder_output': 'mse',
                         'kl_divergence': 'mean_absolute_error',
                         'classification': 'binary_crossentropy'}

    def compile(self, learningRate=0.001, lossWeights=(1.0, 1.0, 1.0), **kwargs):
        """
        Extend tf.keras.Model.compile to work with vae structure and parameters.

        Parameters:
            learningRate : float
                Learning rate to use for the model's optimizer (Adam)
                Default 0.001
            lossWeights : list or tuple
                List of 3 weights for each loss to compute the total loss,
                in order 'decoder_output', 'kl_divergence', 'classification'
                Default (1.0, 1.0, 1.0)
        """

        self.myOptimizer = tf.keras.optimizers.Adam(learning_rate=learningRate)
        self.myLossWeights = {
            'decoder_output': lossWeights[0], 'kl_divergence': lossWeights[1],
            'classification': lossWeights[2]}
        super(vae, self).compile(loss=self.myLosses,
                                 optimizer=self.myOptimizer,
                                 loss_weights=self.myLossWeights, **kwargs)

    def fit(self, jetList, target, validationSplit=0.5, batchSize=800, epochs=30,
            **kwargs):
        """
        Extend tf.keras.Model.fit to work with vae structure and parameters.

        Parameters:
            jetList : numpy 2d array
                Input array of rows of model.encoder_decoder.jetShape jet features
            target : numpy 2d array
                Output targets for the particle classification, in one-hot form.
                Same rows as jetList, model.encoder_decoder.targetShape columns
            validationSplit : float
                Fraction of inputs to use as validation set. Between 0 and 1
                Default 0.5
            batchSize : int
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

        target_kl = np.zeros((jetList.shape[0], 1))
        return super(vae, self).fit(jetList,
                                    {'decoder_output': jetList,
                                     'kl_divergence': target_kl,
                                     'classification': target},
                                    batch_size=batchSize,
                                    validation_split=validationSplit,
                                    epochs=epochs, verbose=2, **kwargs)

    def encoder_predict(self, jetList, **kwargs):
        """
        Generate encoded features and jet type predictions for the input jet features

        Parameters:
            jetList: numpy 2d array
                Input array of rows of model.encoder_decoder.jetShape jet features
        Returns:
            Numpy array(s) of predictions.
        """

        return self.encoder.predict(jetList, **kwargs)

    def decoder_predict(self, encodedFeatures, **kwargs):
        """
        Generate decoded jet predictions for the input encodings and particle types

        Parameters:
            encodedFeatures: numpy 2d array
                Input array of rows of model.encoder_decoder.jetShape encodings
                each with length self.encDimensions + model.encoder_decoder.targetShape
        Returns:
            Numpy array(s) of predictions.
        """

        return self.decoder.predict(encodedFeatures, **kwargs)

    def save(self, customName=''):
        """
        Save vae, encoder and decoder models as three Tensorflow SavedModel
        in the trainPath folder.

        Parameters:
            customName : string
                String to append at each model's folder name
                Default ''
        """

        self.encoder.save(os.path.join(
            trainPath, ''.join(['encoder', customName])))
        self.decoder.save(os.path.join(
            trainPath, ''.join(['decoder', customName])))
        return super(vae, self).save(os.path.join(
            trainPath, ''.join(['vae', customName])))

    def save_weights(self, customName=''):
        """
        Save vae, encoder and decoder model's weights as three .h5 files
        in the trainPath folder.

        Parameters:
            customName : string
                String to append at each model's file name
                Default ''
        """

        self.encoder.save_weights(os.path.join(
            trainPath, ''.join(['encoder', customName, '.h5'])))
        self.decoder.save_weights(os.path.join(
            trainPath, ''.join(['decoder', customName, '.h5'])))
        return super(vae, self).save_weights(os.path.join(
            trainPath, ''.join(['vae', customName, '.h5'])))

    def load_from_file(self, customName=''):
        """
        Load vae, encoder and decoder model's weights from three .h5 files
        in the trainPath folder.

        Parameters:
            customName : string
                String to append at each model's file name
                Default ''

        Returns:
            Loading weights in HDF5 format, so it returns `None`
        """

        self.encoder.load_weights(os.path.join(
            trainPath, ''.join(['encoder', customName, '.h5'])))
        self.decoder.load_weights(os.path.join(
            trainPath, ''.join(['decoder', customName, '.h5'])))
        return super(vae, self).load_weights(os.path.join(
            trainPath, ''.join(['vae', customName, '.h5'])))
