"""Simple pyplot figure saver with some defaults"""

import os
from matplotlib import pyplot as plt

trainPath = os.path.join(os.path.dirname(__file__), '..', 'trained_models', 'figures')

def saveFig(filename, customName='', **kwargs):
    """
    Save current matplotlib.pyplot figure to a .pdf file in the trainPath folder

    Parameters:
        filename : string
            Name of the file for the figure to save
        customName : string
            String to append at each model's file name
            Default ''
    """

    plt.savefig(os.path.join(
        trainPath, ''.join([filename, customName, '.pdf'])), **kwargs)
