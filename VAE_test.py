import os.path

import h5py
import numpy as np
from matplotlib import pyplot as plt

from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Lambda, Reshape
from tensorflow.keras.models import Model
import tensorflow as tf

from sklearn.cluster import Birch

from utilities.file_opener import getJetList

jetList, target = getJetList()

#pt_norm = 100.
#jetList[:, :, 0] = jetList[:, :, 0] / pt_norm
#jetList[:, :, 1:] = jetList[:, :, 1:] * pt_norm

#jetList = jetList[:10000, :, :]
#target = target[:10000, :]

njets = jetList.shape[0]
jet_shape = jetList.shape[1:]
target_shape = target.shape[1:]

encoder_model = tf.keras.models.load_model('Trained_Models/encoder')
decoder_model = tf.keras.models.load_model('Trained_Models/decoder')
autoencoder_model = tf.keras.models.load_model('Trained_Models/autoencoder')

encoded_features = encoder_model.predict(jetList)

colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'purple'}

brc = Birch(branching_factor=50, n_clusters=target_shape[0], threshold=0.1)
brc.fit(encoded_features)

part_pred = brc.predict(encoded_features)
part_real = np.argmax(target, axis=1)

print(part_pred[0:20])
print(part_real[0:20])

plt.figure(1)
plt.scatter(encoded_features[:, 0], encoded_features[:, 1], c=[
            colors[i] for i in part_pred], alpha=0.7)
plt.axis('equal')

plt.figure(2)
plt.scatter(encoded_features[:, 0], encoded_features[:, 1], c=[
            colors[i] for i in part_real], alpha=0.7)
plt.axis('equal')

decoded_jets = decoder_model.predict(encoded_features)

jet_index = 1
print(f'disp. particle color: {colors[part_real[jet_index]]}')

# tmpx = np.arange(jet_shape[0]+1)
# tmpy = np.arange(jet_shape[1]+1)
# tmpx, tmpy = np.meshgrid(tmpx, tmpy)
#
# plt.figure(3)
# #plt.imshow(jetList[jet_index, :, :], origin='lower', cmap='viridis_r',
# #           extent=[0, jet_shape[0], 0, jet_shape[1]])
# plt.pcolormesh(tmpx, tmpy, jetList[jet_index, :, :], cmap='viridis_r', shading='flat')
# plt.colorbar()
#
# plt.figure(4)
# #plt.imshow(decoded_jets[jet_index, :, :], origin='lower', cmap='viridis_r',
# #           extent=[0, jet_shape[0], 0, jet_shape[1]])
# plt.pcolormesh(tmpx, tmpy, decoded_jets[jet_index, :, :], cmap='viridis_r', shading='flat')
# plt.colorbar()


def jet_histogram2d(jet, nbins=100):
    pt = jet[:, 0]
    etarel = jet[:, 1]
    phirel = jet[:, 2]
    newjetImage, xedges, yedges = np.histogram2d(
        etarel, phirel, [nbins, nbins], weights=pt)
    return newjetImage, [xedges[0], xedges[-1], yedges[0], yedges[-1]]


hist_real, extent_real = jet_histogram2d(jetList[jet_index, :, :])
plt.figure(5)
plt.imshow(hist_real, origin='lower', cmap='viridis_r', extent=extent_real)
plt.colorbar()

hist_pred, extent_pred = jet_histogram2d(decoded_jets[jet_index, :, :])
plt.figure(6)
plt.imshow(hist_pred, origin='lower', cmap='viridis_r', extent=extent_pred)
plt.colorbar()

target_kl = np.zeros((njets, 1))
eval_loss = autoencoder_model.evaluate(
    jetList, {'decoder_output': jetList, 'kl_divergence': target_kl})
'''
correct = (part_pred == part_real)
prediction_accuracy = np.count_nonzero(correct) / njets
print(f'pred acc. = {prediction_accuracy * 100 : .2f} %')

njets_per_type = np.count_nonzero(target, axis=0)
pred_acc_per_type = np.array([
        np.count_nonzero(np.logical_and(correct, part_real == i)) / n
        for i, n in enumerate(njets_per_type)])
for i, pacc in enumerate(pred_acc_per_type):
    print(f'particle {i} pred acc. = {pacc * 100 : .2f} %')
'''
plt.show()
