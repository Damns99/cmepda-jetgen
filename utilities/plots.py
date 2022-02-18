"""Plot utility module to help analisys"""

import os
from matplotlib import pyplot as plt
import numpy as np

train_path = os.path.join(os.path.dirname(__file__), '..', 'trained_models',
                          'figures')


def save_figure(filename, custom_name='', **kwargs):
    """
    Save current matplotlib.pyplot figure to a .pdf file in the trainPath
    folder.

    Parameters:
        filename : string
            Name of the file for the figure to save
        custom_name : string
            String to append at each model's file name
            Default ''
    """

    plt.savefig(os.path.join(train_path,
                             ''.join([filename, custom_name, '.pdf'])),
                **kwargs)


def history_plot(history):
    """
    Plot the training history (values of loss and validation loss at each epoch).

    Parameters:
        history : 'History' object
            as returned by model.vae.vae.fit
    """

    plt.plot(history.history["loss"], label='Loss')
    plt.plot(history.history["val_loss"], label='Validation loss')
    plt.yscale('log')
    plt.grid()
    plt.xlabel('Epochs')
    plt.legend()


def jet_scatter(encoded_features, jet_tag, idx1=0, idx2=1):
    """
    Plot a 2d scatter plot of the encoded features, coloured by jet type.

    Parameters:
        encoded_features : numpy 2d array
            features in the encoded space
        jet_tag : numpy 1d array
            jet type array, possible values are integers in range [0,5)
        idx1 : int
            index of the encoded dimension to plot as x axis
            Default 0
        idx2 : int
            index of the encoded dimension to plot as y axis
            Default 1
    """

    colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'purple'}
    plt.scatter(encoded_features[:, idx1], encoded_features[:, idx2], c=[
                colors[i] for i in jet_tag], alpha=0.5, marker='.')
    plt.axis('equal')


def jet_scatter3D(encoded_features, jet_tag, idx1=0, idx2=1, idx3=2):
    """
    Plot a 3d scatter plot of the encoded features, coloured by jet type.

    Parameters:
        encoded_features : numpy 2d array
            features in the encoded space
        jet_tag : numpy 1d array
            jet type array, possible values are integers in range [0,5)
        idx1 : int
            index of the encoded dimension to plot as x axis
            Default 0
        idx2 : int
            index of the encoded dimension to plot as y axis
            Default 1
        idx3 : int
            index of the encoded dimension to plot as z axis
            Default 2
    """

    colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'purple'}
    plt.axes(projection='3d').scatter3D(
        encoded_features[:, idx1], encoded_features[:, idx2],
        encoded_features[:, idx3], c=[colors[i] for i in jet_tag],
        alpha=0.5, marker='.')


def jet_hist(jet_list, feature_names, custom_name='', **kwargs):
    """
    Plot a separate histogram for each jet feature and save them in .pdf format

    Parameters:
        jet_list: numpy 2d array
            Input array of rows of model.encoder_decoder.jetShape jet features
        feature_names : list of strings
            Names of the features for the x axis of each histogram
        custom_name : string
            String to append to the at featureNames to determine each figure's file name
            Default ''
    """

    for name, feature in zip(feature_names, np.transpose(jet_list)):
        plt.figure()
        plt.hist(feature, density=False, **kwargs)
        plt.xlabel(name)
        save_figure(name, custom_name)
