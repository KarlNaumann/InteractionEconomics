import os

import matplotlib as mpl
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from cycler import cycler

PAGE_WIDTH = 5.95114

# Utility functions


def plot_settings():
    """ Set the parameters for plotting such that they are consistent across
    the different models
    """
    mpl.rc('figure', figsize=(PAGE_WIDTH, 6))

    # General Font settings
    x = r'\usepackage[bitstream-charter, greekfamily=default]{mathdesign}'
    mpl.rc('text.latex', preamble=x)
    mpl.rc('text', usetex=True)
    mpl.rc('font', **{'family': 'serif'})

    # Font sizes
    base = 12
    mpl.rc('axes', titlesize=base)
    mpl.rc('legend', fontsize=base-2)
    mpl.rc('axes', labelsize=base-2)

    # Axis styles
    cycles = cycler('linestyle', ['-', '--', ':', '-.'])
    cmap = mpl.cm.get_cmap('tab10')
    cycles += cycler('color', cmap([0.05, 0.15, 0.25, 0.35]))
    mpl.rc('axes', prop_cycle=cycles)


def time_series_plot(df: pd.DataFrame, ax, xtxt: str = '', ytxt: str = '',
                     legend: bool = True):
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

    for series in df.columns:
        ax.plot(df.loc[:, series], label=series)

    if len(df.columns) > 1 and legend:
        ax.legend(ncol=len(df.columns))
    if xtxt == '':
        try:
            ax.set_xlabel(''.join(df.index.names))
        except TypeError:
            try:
                ax.set_xlabel(df.index.name)
            except TypeError:
                pass
    else:
        ax.set_xlabel(xtxt)

    ax.set_ylabel(ytxt)

    ax.set_xlim(df.index[0], df.index[-1])
    ax.minorticks_on()

    return ax


def boundary_layer_approximation(t_end: float, b: float, eps: float,
                                 rho: float, tau_y: float, lam: float,
                                 dep: float):
    """ Calculate the path of production for the classic Solow case based on
    the approximate solution from the boundary layer technique

    Parameters
    ----------
    t_end   :   float
        duration of the simulation
    b   :   float
        constant of integration
    eps :   float
        technology growth rate
    rho :   float
        capital share in cobb-douglas production
    tau_y   :   float
        characteristic timescale of production
    lam :   float
        household saving rate
    dep :   float
        depreciation rate

    Returns
    -------
    solution    :    np.ndarray
        solution path of production

    """
    rhoi = 1 - rho
    constant = (lam / dep) ** (rho / rhoi)
    t = np.arange(int(t_end))
    temp = (b * np.exp(-rhoi * t / tau_y) + 1) ** (1 / rhoi)
    temp += np.exp(eps * t / rhoi)
    return constant * (temp - 1)


def classic_solow_growth_path(t_end: float, start: list, eps: float,
                              rho: float, tau_y: float, lam: float,
                              dep: float):
    """ Function to integrate the path of capital and production in the classic
    solow limiting case

    Parameters
    ----------
    t_end   :   float
        total time of the simulation
    start   :   list
        initial values y0, k0
    eps :   float
        technology growth rate
    rho :   float
        capital share in cobb-douglas production
    tau_y   :   float
        characteristic timescale of production
    lam :   float
        household saving rate
    dep :   float
        depreciation rate

    Returns
    -------
    path    :   pd.DataFrame
        path of capital and production
    """

    path = np.empty((int(t_end), 2))
    path[0, :] = start
    for t in range(1, path.shape[0]):
        y, k = path[t - 1, :]
        v_y = np.exp((rho * k) + (eps * t) - y) - 1
        v_k = lam * np.exp(y - k) - dep
        path[t, 0] = path[t-1, 0] + v_y / tau_y
        path[t, 1] = path[t-1, 1] + v_k
    return path


def output_plot(data: pd.DataFrame, save: str = ''):

    fig = plt.figure(figsize=(PAGE_WIDTH, 3))
    ax = fig.add_subplot()

    labels = dict(ytxt='Production', xtxt='Time')
    time_series_plot(data, ax, **labels)

    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.legend(ncol=len(data.columns), loc=4)

    # Inset axis to highlight the adjustment period
    axins = ax.inset_axes([0.1, 0.5, 0.47, 0.47])
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec='0.8', linestyle='--')
    time_series_plot(data.iloc[int(2e3):int(2e4), :], axins, legend=False)

    fig.tight_layout()

    if save != '':
        if '.png' not in save:
            save += '.png'
        plt.savefig(save, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':

    plot_settings()

    # Parameters (only for lam <= dep)
    # p = dict(rho=0.5, eps=1/10000, tau_y=100, lam=0.15, dep=2e-3)
    p = dict(rho=1/3, eps=1/100000, tau_y=1000, lam=0.2, dep=0.02)
    const = 1.5
    t_end = 1e5

    bla = boundary_layer_approximation(t_end, const, **p)

    # Starting values are in real terms and match the BLA
    ln_y0 = np.log(bla[0])
    ln_k0 = ln_y0 / p['rho']

    solow = classic_solow_growth_path(t_end, [ln_y0, ln_k0], **p)

    data_y = np.hstack([
        bla[:, np.newaxis], np.exp(solow[:, 0, np.newaxis])
    ])

    comparison_y = pd.DataFrame(data_y,
                                columns=['Boundary Layer Approx.', 'Solow'])

    folder = os.getcwd().split('/')
    save = '/'.join(folder[:-1] + ['Paper', 'figures', 'fig_limitks.png'])

    output_plot(comparison_y, save=save)
