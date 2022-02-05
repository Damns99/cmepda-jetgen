from matplotlib import pyplot as plt
import numpy as np


def historyPlot(history):
    plt.plot(history.history["loss"], label='Loss')
    plt.plot(history.history["val_loss"], label='Validation loss')
    plt.yscale('log')
    plt.grid()
    plt.xlabel('Epochs')
    plt.tick_params(direction='in')
    plt.legend()
    plt.show()


def jetHist2D(jetList, nbins=100):
    pt = jetList[:, 0]
    etarel = jetList[:, 1]
    phirel = jetList[:, 2]
    jetImage, xedges, yedges = np.histogram2d(etarel, phirel, [nbins, nbins],
                                              weights=pt)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(jetImage, origin='lower', cmap='viridis_r', extent=extent)
    plt.xlabel(r'$\eta_{rel}$')
    plt.ylabel(r'$\phi_{rel}$')
    plt.colorbar().set_label(r'$p_t$', rotation=0)
    plt.show()


def jetScatter(encoded_features, jetTag):
    colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'purple'}
    plt.scatter(encoded_features[:, 0], encoded_features[:, 1], c=[
                colors[i] for i in jetTag], alpha=0.7)
    plt.axis('equal')
