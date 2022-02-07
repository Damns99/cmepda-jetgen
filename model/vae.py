import os
import numpy as np
import tensorflow as tf
from model.encoder_decoder import (
    encoder_model, encoder_input, decoder_model, decoder_output, kl_divergence)

trainPath = os.path.join(os.path.dirname(__file__), '..', 'trained_models')


class vae(tf.keras.Model):
    def __init__(self):
        super(vae, self).__init__(inputs=encoder_input,
                                  outputs=[decoder_output, kl_divergence],
                                  name='autoencoder')
        self.encoder = encoder_model
        self.decoder = decoder_model
        self.myLosses = {'decoder_output': 'mse',
                         'kl_divergence': 'mean_absolute_error'}

    def compile(self, learningRate=0.001, lossWeights=[1.0, 1.0], **kwargs):
        self.myOptimizer = tf.keras.optimizers.Adam(
            learning_rate=learningRate)
        self.lossWeights = {
            'decoder_output': lossWeights[0], 'kl_divergence': lossWeights[1]}
        super(vae, self).compile(loss=self.myLosses,
                                 optimizer=self.myOptimizer,
                                 loss_weights=self.lossWeights, **kwargs)

    def fit(self, jetList, validationSplit=0.5, batchSize=800, epochs=30,
            **kwargs):
        target_kl = np.zeros((jetList.shape[0], 1))
        return super(vae, self).fit(jetList,
                                    {'decoder_output': jetList,
                                     'kl_divergence': target_kl},
                                    batch_size=batchSize,
                                    validation_split=validationSplit,
                                    epochs=epochs, verbose=2, **kwargs)

    def encoder_predict(self, jetList, **kwargs):
        return self.encoder.predict(jetList, **kwargs)

    def decoder_predict(self, encodedFeatures, **kwargs):
        return self.decoder.predict(encodedFeatures, **kwargs)

    def save(self, customName=''):
        self.encoder.save(os.path.join(trainPath, 'encoder' + customName))
        self.decoder.save(os.path.join(trainPath, 'decoder' + customName))
        return super(vae, self).save(os.path.join(trainPath, 'vae' + customName))

    def save_weights(self, customName=''):
        self.encoder.save_weights(os.path.join(trainPath, 'encoder' + customName + '.h5'))
        self.decoder.save_weights(os.path.join(trainPath, 'decoder' + customName + '.h5'))
        return super(vae, self).save_weights(os.path.join(trainPath, 'vae' + customName + '.h5'))

    def load_from_file(self, customName=''):
        self.encoder.load_weights(os.path.join(trainPath, 'encoder' + customName + '.h5'))
        self.decoder.load_weights(os.path.join(trainPath, 'decoder' + customName + '.h5'))
        return super(vae, self).load_weights(os.path.join(trainPath, 'vae' + customName + '.h5'))

