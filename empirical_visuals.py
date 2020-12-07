import os

import matplotlib as mpl
import numpy as np
import pandas as pd
import pandas_datareader.data as web

from cycler import cycler
from matplotlib import pyplot as plt

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


def load_recession_indicators(startdate: str):
    """ Load the recession boolean series

    Parameters
    ----------
    startdate  :   str

    Returns
    ----------
    data    :   pd.Series
    """
    return web.DataReader('USREC', 'fred', startdate)


def recession_shading(ds: pd.Series, ax, end_date):
    """returns list of (startdate,enddate) tuples for recessions"""
    start, end = [], []
    # Check to see if we start in recession
    if ds.iloc[0, 0] == 1:
        start.extend([ds.index[0]])
    # add recession start and end dates
    for i in range(1, ds.shape[0]):
        a = ds.iloc[i-1, 0]
        b = ds.iloc[i, 0]
        if a == 0 and b == 1:
            start.extend([ds.index[i]])
        elif a == 1 and b == 0:
            end.extend([ds.index[i-1]])
    # if there is a recession at the end, add the last date
    if len(start) > len(end):
        end.extend([ds.index[-1]])

    for j in zip(start, end):
        if j[0] < end_date and j[1] < end_date:
            ax.axvspan(xmin=j[0], xmax=j[1], color='gainsboro')

    return ax


def load_fred_data(tickers: list, start: str) -> dict:
    """ Load all tickers from teh FRED database

    Parameters
    ----------
    tickers  :   list

    Returns
    ----------
    data    :   dict
        dictionary of pd.DataFrame indexed by dataseries name
    """
    return {t: web.DataReader(t, 'fred', start) for t in tickers}


def merge_datasets(data: dict, how='inner'):
    """Merge the data into one dataframe
    Parameters
    ----------
    data  :   dict

    Returns
    ----------
    df    :   pd.DataFrame
    """
    datasets = list(data.keys())
    df = data[datasets[0]]
    for i in datasets[1:]:
        df = df.merge(data[i], left_index=True, right_index=True)
    return df


def autocorrelation(x: pd.Series, lag: int) -> float:
    """ Autocorrelation function for a specific lag

    Parameters
    ----------
    x   :   pd.Series
    lag :   int

    Returns
    ----------
    autocorr    :   float
    """
    x_ = x - x.mean()
    return (x_ * x_.shift(-lag)).mean() / x_.var()


def autocorrelation_function(x: pd.Series, maxlag: int):
    """ Autocorrelation function up to a maximum lag

    Parameters
    ----------
    x   :   pd.Series
    lag :   int

    Returns
    ----------
    autocorr    :   float
    """
    z = np.array([autocorrelation(x, i) for i in range(1, maxlag+1)])
    x = pd.DataFrame(z, index=np.arange(len(z)), columns=['Autocorrelation'])
    return z


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

    for series in df.columns:
        ax.plot(df.loc[:, series], label=series)

    if len(df.columns) > 1:
        ax.legend(ncol=len(df.columns))
    if xtxt == '':
        try:
            ax.set_xlabel(''.join(df.index.names))
        except KeyError:
            try:
                ax.set_xlabel(df.index.name)
            except KeyError:
                pass
    else:
        ax.set_xlabel(xtxt)

    ax.set_ylabel(ytxt)

    ax.set_xlim(df.index[0], df.index[-1])
    ax.minorticks_on()

    return ax


def autocorrelation_plot(df: pd.DataFrame, max_lag: int, ax, xtxt: str = '',
                         ytxt: str = ''):
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

    positions = np.arange(max_lag)
    width = 0.25

    for i, series in enumerate(df.columns):
        autocorr = autocorrelation_function(df.loc[:, series], max_lag)
        pos = [p + i*width for p in positions]
        ax.bar(pos, autocorr, label=series, width=width)

    if len(df.columns) > 1:
        ax.legend()

    if len(df.columns) > 1:
        ax.legend()
    if xtxt == '':
        try:
            ax.set_xlabel(''.join(df.index.names))
        except KeyError:
            try:
                ax.set_xlabel(df.index.name)
            except KeyError:
                pass
    else:
        ax.set_xlabel(xtxt)

    ax.set_ylabel(ytxt)

    return ax


def timeseries_plot(data: pd.DataFrame, rec: bool = True, save: str = ''):

    fig = plt.figure(figsize=(PAGE_WIDTH, 3))
    ax = fig.add_subplot()
    labels = dict(ytxt=r'Quarterly Growth (\%)', xtxt='Time')
    time_series_plot(data, ax, **labels)

    if rec:
        rec = load_recession_indicators(data.index[0].strftime('%Y-%m-%d'))
        recession_shading(rec, ax, data.index[-1])

    if save == '':
        plt.show()
    else:
        if '.png' not in save:
            save += '.png'
        plt.savefig(save, bbox_inches='tight')


if __name__ == '__main__':

    plot_settings()

    timeseries = {
        'GPDIC1': 'Investment',
        'PCECC96': 'Consumption',
        'GCEC1': 'Government',
    }

    data = load_fred_data(list(timeseries.keys()), start='1980-01-01')
    data = merge_datasets(data)
    growth_rates = 100 * data.pct_change().iloc[1:, :]
    growth_rates.columns = [timeseries[i] for i in growth_rates.columns]

    folder = os.getcwd().split('/')
    save = '/'.join(folder[:-1] + ['Paper', 'figures', 'fig_empirics.png'])

    timeseries_plot(growth_rates, rec=True, save=save)
