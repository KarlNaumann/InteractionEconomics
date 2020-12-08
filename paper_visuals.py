import os
import pickle

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib import rc

# Utility functions
def plot_settings():
    """ Set the parameters for plotting such that they are consistent across
    the different models
    """
    sns.set()
    plt.rcParams['text.latex.preamble']\
        = r'\usepackage[bitstream-charter, greekfamily=default]{mathdesign}'
    rc('text', usetex=True)
    rc('font', **{'family': 'serif'})
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams.update({'figure.figsize': (8, 6),
                         'axes.titlesize': 18,
                         'legend.fontsize': 18,
                         'axes.labelsize': 20})

def load_data_from_files(folder:str = 'data/') -> dict:
    """ Function to load the datasets from .csv files in folder

    Parameters
    ----------
    folder  :   str

    Returns
    ----------
    data    :   dict
    """


def load_data_from_FRED():
