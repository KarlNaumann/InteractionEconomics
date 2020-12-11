import os

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import utilities as ut

from matplotlib import pyplot as plt
from statsmodels.tsa.filters.hp_filter import hpfilter


def recession_shading(ax, start_date, end_date):
    """returns list of (startdate,enddate) tuples for recessions"""

    start = start_date.strftime('%Y-%m-%d')
    indicator = web.DataReader('USREC', 'fred', start)

    start, end = [], []
    # Check to see if we start in recession
    if indicator.iloc[0, 0] == 1:
        start.extend([indicator.index[0]])
    # add recession start and end dates
    for i in range(1, indicator.shape[0]):
        a = indicator.iloc[i-1, 0]
        b = indicator.iloc[i, 0]
        if a == 0 and b == 1:
            start.extend([indicator.index[i]])
        elif a == 1 and b == 0:
            end.extend([indicator.index[i-1]])
    # if there is a recession at the end, add the last date
    if len(start) > len(end):
        end.extend([indicator.index[-1]])

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
    data    :   pd.DataFrame
    """
    data = web.DataReader(tickers[0], 'fred', start)
    try:
        # If more dataframes - merge onto the first for matching dates
        for t in tickers[1:]:
            df = web.DataReader(t, 'fred', start)
            data = data.merge(df, left_index=True, right_index=True)
    except KeyError:
        pass

    return data


def oecd_quarterly_gdp(path: str) -> pd.DataFrame:
    """ Read the downloaded OECD dataset on quarterly GDP

    Parameters
    ----------
    path    :   str
        path to the file, normally named 'data/QNA_10122020122218518.csv'

    Returns
    ----------
    data    :   pd.DataFrame
        indexed by date, columns are countries
    """
    data = pd.read_csv(path, usecols=[1, 8, 16])
    data = data.pivot(index='TIME', columns='Country')
    data.columns = data.columns.get_level_values(1)
    data.index = pd.PeriodIndex(data.index, freq='Q').to_timestamp()
    return data


def oecd_industrial_production(path: str) -> pd.DataFrame:
    """ Read the downloaded OECD dataset on monthly industrial production

    Parameters
    ----------
    path    :   str
        path to the file, normally named 'data/DP_LIVE_10122020161421662.csv'

    Returns
    ----------
    data    :   pd.DataFrame
        indexed by date, columns are countries
    """
    data = pd.read_csv('data/DP_LIVE_10122020161421662.csv', usecols=[0, 5, 6])
    data = data.pivot(index='TIME', columns='LOCATION')
    data.columns = data.columns.get_level_values(1)
    data.index = pd.to_datetime(data.index, format='%Y-%m')
    data = data.astype(float).dropna(axis=0)
    return data


def hp_filter(df: pd.DataFrame, lam: int = 1600):
    """ Apply the Hodrick-Prescott filter to each series in the dataframe and
    return the result as a dataframe of cycles and a dataframe of trends

    Parameters
    ----------
    df  :   pd.DataFrame
    lam     :   int

    Returns
    ----------
    cycles  :   pd.DataFrame
    trends  :   pd.DataFrame
    """
    hp = [hpfilter(df.loc[:,series], lamb=lam) for series in df.columns]
    cycles = pd.concat([v[0] for v in hp], axis=1)
    cycles.columns = df.columns
    trends = pd.concat([v[1] for v in hp], axis=1)
    trends.columns = df.columns
    return cycles, trends


def timeseries_plot(data: pd.DataFrame, recessions: bool = True,
                    save: str = '', ytxt: str = r'Quarterly Growth (\%)'):
    """ Generic time-series plot using the predefined timeseries setting
    from utilities and with an option to include U.S. Recession shading

    Parameters
    ----------
    data    :   pd.DataFrame
    recessions  :   bool
    save    :   str
    ytxt    :   str
    """
    fig = plt.figure(figsize=(ut.page_width(), 3))
    ax = fig.add_subplot()
    ut.time_series_plot(data, ax, ytxt=ytxt, xtxt='Time')

    if recessions:
        recession_shading(ax, data.index[0], data.index[-1])

    if save == '':
        plt.show()
    else:
        if '.png' not in save:
            save += '.png'
        plt.savefig(save, bbox_inches='tight')


if __name__ == '__main__':

    # ut.plot_settings(cycles=False)

    folder = os.getcwd().split('/')
    folder = '/'.join(folder[:-1] + ['Paper', 'figures/'])

    # GDP Quarterly data and IP Monthly data

    data_gdp = oecd_quarterly_gdp('data/QNA_10122020122218518.csv')
    data_gdp = data_gdp.loc[:, ['Germany','France','United States']]
    data_ip = oecd_industrial_production('data/DP_LIVE_10122020161421662.csv')
    data_ip = data_ip.loc[:, ['DEU','FRA','USA']]

    # Y-O-Y percentage growth rates

    timeseries_plot(data_gdp.pct_change(4),
                   save=folder+'fig_emp_gdp_yoy.png', ytxt=r'GDP y-o-y')

    timeseries_plot(data_ip.pct_change(12),
                   save=folder+'fig_emp_ip_yoy.png', ytxt=r'IP y-o-y')

    # Hodrick-Prescott Filter

    cycle, _ = hp_filter(data_gdp, lam=1600)
    timeseries_plot(cycle, recessions=False, ytxt=r'GDP HP Cycle',
                    save=folder+'fig_emp_gdp_hpfilter.png')

    cycle, _ = hp_filter(data_ip, lam=129600)
    timeseries_plot(cycle, recessions=False, ytxt=r'IP HP Cycle',
                    save=folder+'fig_emp_ip_hpfilter.png')

    # Bandpass Filter approach on log-growth rates
    # Can be done with from statsmodels.tsa.filters.bk_filter import bkfilter
