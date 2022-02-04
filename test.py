import numpy as np
from model.vae import vae
from utilities.file_opener import getJetList

learning_rate = 0.001
epochs = 30
batch_size = 800
lossWeights = [1.0, 0.1]


jetList, _ = getJetList()

vae = vae()
vae.summary()
vae.compile()
# history = vae.fit(jetList)
