import numpy as np
from matplotlib import pyplot as plt
from model.vae import vae
from utilities.file_opener import getJetList
from utilities.plots import historyPlot, jetScatter, jetHist2D

learning_rate = 0.001
lossWeights = [1.0, 0.1]
epochs = 5
batch_size = 50
validation_split = 0.5

jetList, target = getJetList(test=True)

vae = vae()
vae.summary()
vae.compile(learning_rate=learning_rate, lossWeights=lossWeights)
history = vae.fit(jetList, validation_split=validation_split,
                  batch_size=batch_size, epochs=epochs)
historyPlot(history)
vae.save()

jetTag = np.argmax(target, axis=1)

encoded_features = vae.encoder_predict(jetList)
decoded_jets = vae.decoder_predict(encoded_features)

plt.figure()
jetScatter(encoded_features, jetTag)

plt.figure()
jetHist2D(jetList[1, :, :])

plt.figure()
jetHist2D(decoded_jets[1, :, :])

plt.show()
