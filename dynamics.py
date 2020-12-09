import os
import pickle

import numpy as np
import pandas as pd
import utilities as ut

from matplotlib import pyplot as plt
from scipy import stats as ss


def file_category(folder: str = 'simulations', t_end: str = 't1e+07'):
    """Generate a dataframe containing info on all the files in the folder"""

    files = [f for f in os.listdir(folder) if '.df' in f]

    p = ut.filename_extraction(files[0], seed=True)
    df = pd.DataFrame(index=list(range(len(files))),
                      columns=list(p.keys()) + ['file'])

    for i, f in enumerate(files):
        df.iloc[i, :-1] = ut.filename_extraction(f, seed=True)
        df.loc[i, 'file'] = folder + '/' + f

    df.set_index('file', inplace=True)
    return df


def cycle_timing(gdp, timescale=63):
    gdp_growth = gdp.copy(deep=True).iloc[::timescale].pct_change()
    ix = gdp_growth < 0
    # expansion start: negative growth to two consecutive expansions
    expansions = np.flatnonzero(
            (ix.shift(-1) == False) & (ix == True) & (ix.shift(1) == True))
    # recession start: positive growth to two consecutive contractions
    recessions = np.flatnonzero(
            (ix == True) & (ix.shift(1) == False) & (ix.shift(2) == False))

    expansions = [gdp_growth.index[i] for i in expansions]
    recessions = [gdp_growth.index[i] for i in recessions]

    se = [(0, min(recessions))]
    i = 0

    while i < len(expansions):
        # Determine the index of the next recession
        rec_ix = min([ix for ix in recessions if ix > expansions[i]]
                     + [gdp.shape[0]])
        # Check if the recession is admissible
        if rec_ix > se[-1][1]:
            se.append((expansions[i], rec_ix))
        # Find expansion after this
        exp = min([i for i in expansions if i > rec_ix] + [gdp.shape[0]])
        if exp == gdp.shape[0]:
            break
        else:
            i = expansions.index(exp)

    return pd.DataFrame(se[1:], columns=['expansion', 'recession'])


def cycle_analysis(paths):
    dfs = []
    for path in paths:
        analysis = cycle_timing(path.y, timescale=63)
        # Business cycle duration = expansion to expansion distance
        duration = analysis.expansion.diff()
        analysis.loc[:, 'duration'] = 0
        analysis.iloc[:-1, -1] = duration.iloc[1:].values
        # Peak, Trough & Depth
        analysis.loc[:, 'peak'] = 0
        analysis.loc[:, 'trough'] = 0
        for i in analysis.index[:-1]:
            start = analysis.expansion.iloc[i]
            end = analysis.expansion.iloc[i + 1]
            prod = path.y.loc[start:end]
            peak, peak_ix = prod.max(), prod.idxmax()
            trough = path.y.loc[peak_ix:end].min()
            analysis.loc[i, ['peak', 'trough']] = peak, trough

        analysis = analysis.iloc[:-1]
        analysis.loc[:, 'p2t'] = analysis.peak - analysis.trough
        analysis.loc[:, 'p2t_adj'] = analysis.p2t / analysis.duration
        dfs.append(analysis)

    return dfs, pd.concat(dfs, axis=0)


def histogram(series: pd.Series, ax, win: bool = True, bins: int = 20,
              xtxt: str = '', ytxt: str = 'Proportion'):
    """ Function to generically plot a histogram

    Parameters
    ----------
    Series  :   pd.Series
    ax  :   matplotlib axis object
    win :   bool
    xtxt    :   str
    ytxt    :   str
    """

    x = winsorize(pd.Series) if win else pd.Series
    n, bins = np.histogram(x, bins=bins)
    n = n / pd.Series.shape[0]

    ax.hist(bins[:-1], bins, weights=n)
    ax.set_xlabel(xtxt)
    ax.set_ylabel(ytxt)


def duration_depth_histogram(df: pd.DataFrame, save=''):

    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(ut.page_witdh(), 3)

    histogram(df.duration / 250, ax[0], xtxt='Duration (Years)')
    histogram(df.p2t / 250, ax[1], xtxt='Depth (Log production)')

    plt.tight_layout()
    if save != '':
        save = save + '.png' if '.png' not in save else save
        plt.savefig(save, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def winsorize(series, perc=[0.05, 0.95]):
    quint = series.quantile(perc)
    new = series[series > quint.min()]
    return new[new < quint.max()]


def business_cycle_hist_v2(info, files, period, save=''):
    """Histogram of business cycle lengths per gamnma c2 combination"""
    paths = [v for i, v in ut.load_sims(files.index).items()]
    paths = smoothing(paths, period)
    _, analysis = cycle_analysis(paths)

    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(8, 4)

    p2t = winsorize(analysis.p2t)
    n_p2t, bins_p2t = np.histogram(p2t, bins=20)
    n_p2t = n_p2t / p2t.shape[0]

    dur = winsorize(analysis.duration.astype(float) / 250)
    dur.dropna(inplace=True)
    dur = dur[dur!=np.inf]
    print(dur.values)
    n, bins, _ = ax[0].hist(dur, bins=20, density=True)
    par = ss.fit_truncnorm(dur.values, bins[0], bins[-2])

    x = np.linspace(bins[0], bins[-2], 100)
    ax[0].plot(x, ss.truncnorm.pdf(x, *par))
    ax[0].set_xlabel('Duration (Years)')
    ax[0].set_ylabel('Proportion')

    ax[1].hist(bins_p2t[:-1], bins_p2t, weights=n_p2t)
    # ax[1].hist(winsorize(analysis_df.p2t), **args)
    ax[1].set_xlabel('Depth (y)')
    ax[1].set_ylabel('Proportion')
    plt.tight_layout()
    if save != '':
        save = save + '.png' if '.png' not in save else save
        plt.savefig(save, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def business_cycle_histograms(info, files, period, winsorize=False, save=''):
    """Histogram of business cycle lengths per gamnma c2 combination"""
    paths = [v for i, v in ut.load_sims(files.index).items()]
    paths = smoothing(paths, period)
    _, analysis = cycle_analysis(paths)
    duration_depth_histogram(analysis, save=save)


def smoothing(paths: list, period: int):
    """Smooth the timeseries with a moving average across p periods"""
    return [df.rolling(period, min_periods=period).mean() for df in paths]


def plot_all_timeseries(series):
    for i, v in enumerate(series):
        info, files = v
        paths = [v for i, v in ut.load_sims(files.index).items()]
        paths = [df.rolling(T, min_periods=T).mean() for df in paths]
        for i, df in enumerate(paths):
            save = fig_name(FOLDER_GENERAL, files.iloc[i, :], 'timeseries')
            timseries_plot(df, info, stop=int(4e5), save=save)


def timseries_plot(df, info, stop=int(1e6), save=''):
    # General figure of a path
    fig = plt.figure()
    fig.set_size_inches(10, 6)
    gs = fig.add_gridspec(3, 4)
    mask = df.ks >= df.kd
    axs = []

    # Production timeseries
    ax = fig.add_subplot(gs[0, :-1])
    ax.plot(df.y.iloc[:stop])
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.xaxis.major.formatter._useMathText = True
    ax.set_xlabel('Time t (business days)')
    ax.set_ylabel('Log Production y')
    axs.append(ax)

    # Mini-histogram of growth rates
    ax = fig.add_subplot(gs[0, -1])
    ax.hist(df.y.diff()[mask], bins=50)
    # ax.xaxis.set_major_formatter(mticker.FuncFormatter(update_ticks))
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.xaxis.major.formatter._useMathText = True
    ax.xaxis.offsetText.set_fontsize(8)
    ax.set_yticks([], minor=True)
    ax.set_yticks([])
    ax.set_xlabel('Growth y for kd<ks')
    axs.append(ax)

    # Capital timeseries
    ax = fig.add_subplot(gs[1, :-1])
    ax.plot(df.ks.iloc[:stop], label='Supply')
    ax.plot(df.kd.iloc[:stop], label='Demand')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.xaxis.major.formatter._useMathText = True
    ax.legend()
    ax.set_xlabel('Time t (business days)')
    ax.set_ylabel('Log Capital (ks, kd)')
    axs.append(ax)

    # Mini-histogram of growth rates
    ax = fig.add_subplot(gs[1, -1])
    ax.hist(df.loc[:, ['ks', 'kd']].min(axis=1).diff()[mask], bins=50)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.xaxis.major.formatter._useMathText = True
    ax.xaxis.offsetText.set_fontsize(8)
    ax.set_yticks([], minor=True)
    ax.set_yticks([])
    ax.set_xlabel('Growth k for kd<ks')
    axs.append(ax)

    # Sentiment timeseries
    ax = fig.add_subplot(gs[2, :-1])
    ax.plot(df.s.iloc[:stop])
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.xaxis.major.formatter._useMathText = True
    ax.set_xlabel('Time t (business days)')
    ax.set_ylabel('Sentiment s')
    axs.append(ax)

    # Mini-histogram
    ax = fig.add_subplot(gs[2, -1])
    ax.hist(df.s, bins=50)
    ax.set_yticks([], minor=True)
    ax.set_yticks([])
    ax.set_xlabel('Sentiment s')
    axs.append(ax)

    plt.tight_layout()

    if save != '':
        save = save + '.png' if '.png' not in save else save
        plt.savefig(save, bbox_inches='tight')
        plt.close()
    else:
        fig.suptitle('c2 {:.1e} gamma {:.1e}'.format(*info))
        plt.show()


def fig_name(folder, info, kind):
    struct = [
        'g_{:.0f}_'.format(info.gamma),
        'c2_{:.0f}_'.format(info.c2 * 1e5)
    ]

    try:
        struct.append('seed_{:.0f}'.format(info.seed))
    except AttributeError:
        pass

    return '{}fig_{}_{}.png'.format(folder, kind, ''.join(struct))


if __name__ == '__main__':

    ut.plot_settings()

    plt.rcParams['axes.prop_cycle'] = plt.cycler("color", plt.cm.tab10.colors)

    # Folders
    FOLDER_DEMAND = '/Users/karlnaumann/Desktop/Solow_Model/paths_demand/'
    FOLDER_GENERAL = '/Users/karlnaumann/Desktop/Solow_Model/paths_general/'
    T = 2000

    file_df = ut.parse_directory(folder=FOLDER_GENERAL)
    series = file_df.groupby(['c2', 'gamma'])

    for i, v in enumerate(series):
        info, files = v
        info = pd.Series(info, index=['c2', 'gamma'])
        name = fig_name(FOLDER_GENERAL, info, 'histograms')
        business_cycle_hist_v2(info, files, T)
