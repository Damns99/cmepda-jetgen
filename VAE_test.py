import os.path

import h5py
import numpy as np
from matplotlib import pyplot as plt

from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Lambda, Reshape
from tensorflow.keras.models import Model
import tensorflow as tf

jetList = np.array([])
target = np.array([])

datafiles = ['~/Documents/CMEPDA/CMEPDA_exam/Data/jetImage_7_100p_80000_90000.h5']
for fileIN in datafiles:
    f = h5py.File(os.path.expanduser(fileIN))
    # for pT, etarel, phirel [5, 8, 11]
    myJetList = np.array(f.get("jetConstituentList")[:, :, [5, 8, 11]])
    mytarget = np.array(f.get('jets')[:, -6:-1])
    jetList = np.concatenate([jetList, myJetList], axis=0) if jetList.size else myJetList
    target = np.concatenate([target, mytarget], axis=0) if target.size else mytarget
    del myJetList, mytarget
    f.close()

pt_norm = 100.
jetList[:, :, 0] = jetList[:, :, 0] / pt_norm

njets = jetList.shape[0]
jet_shape = jetList.shape[1:]

encoder_model = tf.keras.models.load_model('Trained_Models/encoder')
decoder_model = tf.keras.models.load_model('Trained_Models/decoder')

encoded_features = encoder_model.predict(jetList)
print(np.shape(encoded_features))

colors = {0:'red', 1:'green', 2:'blue', 3:'yellow', 4:'purple'}
target_colors = np.argmax(target, axis=1)

plt.figure(1)
plt.scatter(encoded_features[:, 0], encoded_features[:, 1], c=[colors[i] for i in target_colors])
plt.axis('equal')

decoded_jets = decoder_model.predict(encoded_features)
print(np.shape(decoded_jets))

def jet_histogram2d(jet, nbins=100):
    pt = jet[:, 0]
    etarel = jet[:, 1]
    phirel = jet[:, 2]
    print(np.sum(pt>0))
    print(np.max(etarel), np.min(etarel))
    print(np.max(phirel), np.min(phirel))
    newjetImage = np.histogram2d(etarel, phirel, [nbins, nbins], [[-0.5, 0.5],[-0.5, 0.5]], weights=pt)
    return newjetImage[0]

jet_index = 0
print(colors[target_colors[jet_index]])

plt.figure(2)
plt.imshow(jet_histogram2d(jetList[jet_index, :, :]), origin='lower', cmap='viridis_r')
plt.figure(3)
plt.imshow(jet_histogram2d(decoded_jets[jet_index, :, :]), origin='lower', cmap='viridis_r')

print(np.mean((jetList[jet_index, :, :] - decoded_jets[jet_index, :, :])**2))

plt.show()