"""Open some relevant fields of an axample dataset and print/plot them"""

import os.path
import h5py
import numpy as np
import matplotlib.pyplot as plt


def plotFeature(feature, xlabel):
    """
    Plot an histogram of a given feature and show it

    Parameters:
        feature : numpy array
            Input to plot
        xlabel : string
            Name of the feature for the x axis
    """

    plt.hist(feature, 20)
    plt.xlabel(xlabel)
    plt.show()


filename = 'jetImage_7_100p_30000_40000.h5'
path = os.path.join(os.path.dirname(__file__), '..', 'data', filename)
f = h5py.File(path)

# print(list(f.keys()))

particleFeatureNames = np.array(f.get('particleFeatureNames'))
# print(particleFeatureNames)

jetConstituentList = np.array(f.get('jetConstituentList'))
# print(np.shape(jetConstituentList))
# print(jetConstituentList[0, 0:5, [5, 8, 11]])

jetImage = f.get('jetImage')
# print(jetImage.shape)
# plt.imshow(jetImage[0], cmap='viridis_r')
# plt.colorbar()
# plt.show()

jetFeatureNames = np.array(f.get('jetFeatureNames'))
# print(jetFeatureNames)

jets = np.array(f.get('jets'))
# print(jets.shape)

pt = jetConstituentList[:, :, 5]
etarel = jetConstituentList[:, :, 8]
phirel = jetConstituentList[:, :, 11]

# plotFeature(pt[0, :], r'$p_t$')
# plotFeature(etarel[0, :], r'$\eta_{rel}$')
# plotFeature(phirel[0, :], r'$\phi_{rel}$')
