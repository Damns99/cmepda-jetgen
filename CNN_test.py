import os.path

import h5py
import numpy as np
from matplotlib import pyplot as plt

from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Lambda, Reshape
from tensorflow.keras.models import Model
import tensorflow as tf

jetList = np.array([])
target = np.array([])

datafiles = ['~/Documents/CMEPDA/CMEPDA_exam/Data/jetImage_7_100p_70000_80000.h5']
for fileIN in datafiles:
    f = h5py.File(os.path.expanduser(fileIN))
    # for pT, etarel, phirel [5, 8, 11]
    myJetList = np.array(f.get("jetConstituentList")[:, :, [5, 8, 11]])
    mytarget = np.array(f.get('jets')[:, -6:-1])
    jetList = np.concatenate([jetList, myJetList], axis=0) if jetList.size else myJetList
    target = np.concatenate([target, mytarget], axis=0) if target.size else mytarget
    del myJetList, mytarget
    f.close()

pt_norm = 500.
jetList[:, :, 0] = jetList[:, :, 0] / pt_norm

jetList = jetList[:10000, :, :]
target = target[:10000, :]

njets = jetList.shape[0]
jet_shape = jetList.shape[1:]
target_shape = target.shape[1:]

encoder_model = tf.keras.models.load_model('Trained_Models/cnn')

encoded_features = encoder_model.predict(jetList)
eval_loss = encoder_model.evaluate(jetList, target)

colors = {0:'red', 1:'green', 2:'blue', 3:'yellow', 4:'purple'}

print(encoded_features[0:10, :])
print(target[0:10, :])

part_pred = np.argmax(encoded_features, axis=1)
part_real = np.argmax(target, axis=1)

print(part_pred[0:20])
print(part_real[0:20])

plt.figure(1)
plt.axis('equal')
plt.grid()
plt.subplot(221)
plt.scatter(encoded_features[:, 0], encoded_features[:, 1], c=[colors[i] for i in part_pred], marker='.', alpha=0.7)
plt.subplot(222)
plt.scatter(encoded_features[:, 0], encoded_features[:, 2], c=[colors[i] for i in part_pred], marker='.', alpha=0.7)
plt.subplot(223)
plt.scatter(encoded_features[:, 0], encoded_features[:, 3], c=[colors[i] for i in part_pred], marker='.', alpha=0.7)
plt.subplot(224)
plt.scatter(encoded_features[:, 0], encoded_features[:, 4], c=[colors[i] for i in part_pred], marker='.', alpha=0.7)

plt.figure(2)
plt.axis('equal')
plt.grid()
plt.subplot(221)
plt.scatter(encoded_features[:, 0], encoded_features[:, 1], c=[colors[i] for i in part_real], marker='.', alpha=0.7)
plt.subplot(222)
plt.scatter(encoded_features[:, 0], encoded_features[:, 2], c=[colors[i] for i in part_real], marker='.', alpha=0.7)
plt.subplot(223)
plt.scatter(encoded_features[:, 0], encoded_features[:, 3], c=[colors[i] for i in part_real], marker='.', alpha=0.7)
plt.subplot(224)
plt.scatter(encoded_features[:, 0], encoded_features[:, 4], c=[colors[i] for i in part_real], marker='.', alpha=0.7)

correct = (part_pred == part_real)
prediction_accuracy = np.count_nonzero(correct) / njets
print(f'pred acc. = {prediction_accuracy * 100 : .2f} %')

njets_per_type = np.count_nonzero(target, axis=0)
pred_acc_per_type = np.array([
        np.count_nonzero(np.logical_and(correct, part_real == i)) / n
        for i, n in enumerate(njets_per_type)])
for i, pacc in enumerate(pred_acc_per_type):
    print(f'particle {i} pred acc. = {pacc * 100 : .2f} %')

plt.show()