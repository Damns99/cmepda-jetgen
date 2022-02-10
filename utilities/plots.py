from matplotlib import pyplot as plt
import numpy as np

from utilities.figure_saver import saveFig


def historyPlot(history):
    plt.plot(history.history["loss"], label='Loss')
    plt.plot(history.history["val_loss"], label='Validation loss')
    plt.yscale('log')
    plt.grid()
    plt.xlabel('Epochs')
    plt.tick_params(direction='in')
    plt.legend()


# def jetHist2D(jetList, nbins=100):
#     pt = jetList[:, 0]
#     etarel = jetList[:, 1]
#     phirel = jetList[:, 2]
#     jetImage, xedges, yedges = np.histogram2d(etarel, phirel, [nbins, nbins],
#                                               weights=pt)
#     extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#     plt.imshow(jetImage, origin='lower', cmap='viridis_r', extent=extent)
#     plt.xlabel(r'$\eta_{rel}$')
#     plt.ylabel(r'$\phi_{rel}$')
#     plt.colorbar().set_label(r'$p_t$', rotation=0)


def jetScatter(encoded_features, jetTag, idx1=0, idx2=1):
    colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'purple'}
    plt.scatter(encoded_features[:, idx1], encoded_features[:, idx2], c=[
                colors[i] for i in jetTag], alpha=0.5, marker='.')
    plt.axis('equal')

def jetScatter3D(encoded_features, jetTag, idx1=0, idx2=1, idx3=2):
    colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'purple'}
    plt.axes(projection='3d').scatter3D(encoded_features[:, idx1], encoded_features[:, idx2], encoded_features[:, idx3],  c=[colors[i] for i in jetTag], alpha=0.5, marker='.')

def jetHist(jetList, featureNames, customName='', **kwargs):
    for name, feature in zip(featureNames, np.transpose(jetList)):
        plt.figure()
        plt.hist(feature, density=False, **kwargs)
        plt.xlabel(name)
        saveFig(name, customName)
