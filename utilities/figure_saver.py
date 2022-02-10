import os
from matplotlib import pyplot as plt

path = os.path.join(os.path.dirname(__file__), '..', 'trained_models', 'figures')

def saveFig(filename, customName='', **kwargs):
    plt.savefig(os.path.join(path, ''.join([filename, customName, '.pdf'])))