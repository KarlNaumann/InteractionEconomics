import pickle
import os

import pandas as pd
import numpy as np

from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.cm import get_cmap
from seaborn import heatmap as sns_heatmap


def page_width():
    return 5.95114


def filename_extraction(filename: str, seed: bool = False) -> dict:
    """ Extract the parameter values from the filename and return them in the
    form of a dictionary. Also extracts the simulation duration.

    Parameters
    ----------
    filename    :   str

    Returns
    -------
    parameters  :   dict
    """
    parts = filename[:-3].split('_')
    names = ['tech0', 'rho', 'epsilon', 'saving0', 'dep', 'tau_y', 'tau_s',
             'tau_h', 'c1', 'c2', 'beta1', 'beta2', 'gamma']
    if not seed:
        names.extend(['t_end'])
        loc = [17, 18, 3, 15, 16, 12, 13, 14, 5, 7, 9, 11, 2, 1]
        cut = [4, 3, 1, 3, 3, 2, 2, 2, 0, 0, 0, 0, 1, 1]
    else:
        names.extend(['seed'])
        loc = [17, 18, 3, 15, 16, 12, 13, 14, 5, 7, 9, 11, 2, -1]
        cut = [4, 3, 1, 3, 3, 2, 2, 2, 0, 0, 0, 0, 1, 0]

    return {p[0]: float(parts[p[1]][p[2]:]) for p in zip(names, loc, cut)}


def listdir(path):
    """ List a directory contents, ignoring hidden files

    Parameters
    ----------
    path  :   str

    Returns
    -------
    files   :   list
    """
    return [f for f in os.listdir(path) if not f.startswith('.')]


def parse_directory(folder: str, criteria: list = ['.df'], seed: bool = False):
    """ Combine into dataframe, indexed by filename, the various parameters for
    the simulations that have been run

    Parameters
    ----------
    folder  :   str
    criteria   :   list[str]
        parts of the filename that must be included e.g. '.df', 't_end'

    Returns
    -------
    df  :   pd.DataFrame
    """
    # Filenames that contain all the criteria information
    files = [f for f in listdir(folder) if all([i in f for i in criteria])]
    data = {f: filename_extraction(f, seed) for f in files}
    df = pd.DataFrame(data).T
    df.index = [folder + '/' + i for i in df.index]
    return df


def load_sims(files: list):
    """ Load stored dataframes from their respective pickle files

    Parameters
    ----------
    files   :   list
    t_end   :   str

    Returns
    -------
    sims    :   dict
    """
    sims = {}
    for path in files:
        file = open(path, 'rb')
        df = pickle.load(file)
        sims[path] = df
        file.close()
    return sims


def plot_settings():
    """ Set the parameters for plotting such that they are consistent across
    the different models
    """
    rc('figure', figsize=(page_width(), 6))

    # General Font settings
    x = r'\usepackage[bitstream-charter, greekfamily=default]{mathdesign}'
    rc('text.latex', preamble=x)
    rc('text', usetex=True)
    rc('font', **{'family': 'serif'})

    # Font sizes
    base = 12
    rc('axes', titlesize=base-2)
    rc('legend', fontsize=base-2)
    rc('axes', labelsize=base-2)
    rc('xtick', labelsize=base-3)
    rc('ytick', labelsize=base-3)

    # Axis styles
    cycles = cycler('linestyle', ['-', '--', ':', '-.'])
    cmap = get_cmap('gray')
    cycles += cycler('color', cmap(list(np.linspace(0.1,0.9,4))))
    rc('axes', prop_cycle=cycles)


def c2_gamma_heatmap(df: pd.DataFrame, label: str, save: str = '',
                     show: bool = False, limits: tuple = None, freq=2):
    """ Function to generate a heatmap from the given dataframe,
    including limits for the colorbar

    Parameters
    ----------
    df      :   pd.DataFrame
    ax      :   matplotlib axes object
    label   :   str

    save    :   str
        filepath to save the figure
    show    :   bool
        whether to display the figure
    limits  :   tuple
        tuple of lower and higher limits for the colormapping
    freq    :   int
        frequency for the labels

    Returns
    ----------
    ax  :   matplotlib axes object
    """

    fig = plt.figure(figsize=(page_width(), 4))
    ax = fig.add_subplot()

    # Plot the heatmap using seaborn
    limits = limits if limits is not None else (None, None)
    params = dict(cmap='gray', linewidths=0.1, ax=ax, vmin=limits[0],
                  vmax=limits[1])
    _ = sns_heatmap(df.astype(float), cbar_kws={'label': label}, **params)

    col, ix = df.columns.to_list(), df.index.to_list()

    # X labelling
    ax.set_xticks(np.arange(0, len(col), freq) + 0.5)
    xticks = ['{:.1f}'.format(i * 1e-3) for i in sorted(col)[::freq]]
    ax.set_xticklabels(xticks)
    ax.text(1.05, -0.05, '1e3', transform=ax.transAxes,
            horizontalalignment='right',
            verticalalignment='top')

    # Y labelling
    ax.set_yticks(np.arange(0, len(ix), freq) + 0.5)
    yticks = ['{:.1f}'.format(i * 1e4) for i in sorted(ix)[::freq]]
    ax.set_yticklabels(yticks)
    ax.text(-0.05, 1.0, '1e-4', transform=ax.transAxes,
            horizontalalignment='left',
            verticalalignment='bottom')

    ax.tick_params(axis='x', labelrotation=0)
    ax.invert_yaxis()

    # plt.tight_layout()
    if save != '':
        plt.savefig(save, bbox_inches='tight', format='eps')
    if show:
        plt.show()
    plt.close()


def time_series_plot(df: pd.DataFrame, ax, xtxt: str = '', ytxt: str = ''):
    """ Generate a timeseries graph on the axes for each column in the
    given dataframe

    Parameters
    ----------
    df  :   pd.DataFrame
    ax  :   matplotlib axes object

    Returns
    ----------
    ax  :   matplotlib axes object
    """

    try:
        for series in df.columns:
            ax.plot(df.loc[:, series], label=series)
        if len(df.columns) > 1:
            ax.legend(ncol=len(df.columns))
    except AttributeError:
        ax.plot(df, label = df.name)

    if xtxt == '':
        try:
            ax.set_xlabel(''.join(df.index.names))
        except KeyError:
            try:
                ax.set_xlabel(df.index.name)
            except KeyError:
                pass
        except TypeError: 
            pass

    else:
        ax.set_xlabel(xtxt)

    ax.set_ylabel(ytxt)

    ax.set_xlim(df.index[0], df.index[-1])
    ax.minorticks_on()

    return ax
