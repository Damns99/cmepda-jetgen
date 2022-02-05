import numpy as np
import tensorflow as tf
from model.encoder_decoder import (
    encoder_model, encoder_input, decoder_model, decoder_output, kl_divergence)


class vae(tf.keras.Model):
    def __init__(self):
        super(vae, self).__init__(inputs=encoder_input,
                                  outputs=[decoder_output, kl_divergence],
                                  name='autoencoder')
        self.encoder = encoder_model
        self.decoder = decoder_model
        self.myLosses = {'decoder_output': 'mse',
                         'kl_divergence': 'mean_absolute_error'}

    def compile(self, learning_rate=0.001, lossWeights=[1.0, 1.0], **kwargs):
        self.myOptimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate)
        self.lossWeights = {
            'decoder_output': lossWeights[0], 'kl_divergence': lossWeights[1]}
        super(vae, self).compile(loss=self.myLosses,
                                 optimizer=self.myOptimizer,
                                 loss_weights=self.lossWeights)

    def fit(self, jetList, validation_split=0.5, batch_size=800, epochs=30,
            **kwargs):
        target_kl = np.zeros((jetList.shape[0], 1))
        return super(vae, self).fit(jetList,
                                    {'decoder_output': jetList,
                                     'kl_divergence': target_kl},
                                    batch_size=batch_size,
                                    validation_split=validation_split,
                                    epochs=epochs, verbose=2)

    def predict_encoder(self, jetList, **kwargs):
        return self.encoder.predict(jetList, **kwargs)

    def predict_decoder(self, jetList, **kwargs):
        return self.decoder.predict(jetList, **kwargs)
