import os
import pickle

import numpy as np
import pandas as pd
from scipy import stats as ss
from scipy.optimize import fmin_slsqp
from matplotlib import pyplot as plt
from scipy.stats.mstats import winsorize


def filename_extraction(filename: str) -> dict:
    """ Extract parameters from filename"""
    parts = filename[:-3].split('_')
    params = {}
    variables = ['tech0', 'rho', 'epsilon', 'saving0', 'dep',
                 'tau_y', 'tau_s', 'tau_h', 'c1', 'c2',
                 'beta1', 'beta2', 'gamma']
    locs = [17, 18, 3, 15, 16, 12, 13, 14, 5, 7, 9, 11, 2]
    cuts = [4, 3, 1, 3, 3, 2, 2, 2, 0, 0, 0, 0, 1]

    for p in zip(variables, locs, cuts):
        params[p[0]] = float(parts[p[1]][p[2]:])

    seed = float(parts[-1])

    return params, seed


def file_category(folder: str = 'simulations', t_end: str = 't1e+07'):
    """Generate a dataframe containing info on all the files in the folder"""

    files = [f for f in os.listdir(folder) if '.df' in f]

    p, _ = filename_extraction(files[0])
    df = pd.DataFrame(index=list(range(len(files))),
                      columns=list(p.keys()) + ['seed', 'file'])

    for i, f in enumerate(files):
        df.iloc[i, :-2], df.loc[i, 'seed'] = filename_extraction(f)
        df.loc[i, 'file'] = folder + '/' + f

    df.set_index('file', inplace=True)
    return df


def load_simulation_paths(files: list):
    return [pickle.load(open(p, 'rb')) for p in files]


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


def duration_depth_histogram(analysis_df, save=''):
    dur = winsorize(analysis_df.duration / 250)
    n_dur, bins_dur = np.histogram(dur, bins=20)
    n_dur = n_dur / dur.shape[0]

    p2t = winsorize(analysis_df.p2t)
    n_p2t, bins_p2t = np.histogram(p2t, bins=20)
    n_p2t = n_p2t / p2t.shape[0]

    args = dict(density=True, bins=20)
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(8, 4)
    ax[0].hist(bins_dur[:-1], bins_dur, weights=n_dur)
    # ax[0].hist(winsorize(analysis_df.duration/250), **args)
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


def winsorize(series, perc=[0.05, 0.95]):
    quint = series.quantile(perc)
    new = series[series > quint.min()]
    new = new[new < quint.max()]
    return new


def business_cycle_hist_v2(info, files, period, save=''):
    """Histogram of business cycle lengths per gamnma c2 combination"""
    paths = load_simulation_paths(files.index.to_list())
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
    par = fit_truncnorm(dur.values, bins[0], bins[-2])

    print(par)
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


def fit_truncnorm(data, xa, xb):

    def func(p, data, xa, xb):
        return ss.truncnorm.nnlf(p, data)

    def constraint(p, r, xa, xb):
        a, b, loc, scale = p
        return np.array([a*scale+loc-xa, b*scale + loc - xb])

    mu0, sig0 = np.mean(data), np.std(data)
    a0, b0 = (xa-mu0)/sig0, (xb-mu0)/sig0
    p0 = [a0, b0, mu0, sig0]
    par = fmin_slsqp(func, p0, f_eqcons=constraint, args=(data, xa, xb),
                     iprint=False, iter=1000)

    return par

def business_cycle_histograms(info, files, period, winsorize=False, save=''):
    """Histogram of business cycle lengths per gamnma c2 combination"""
    paths = load_simulation_paths(files.index.to_list())
    paths = smoothing(paths, period)
    _, analysis = cycle_analysis(paths)
    duration_depth_histogram(analysis, save=save)


def smoothing(paths, period):
    """Smooth the timeseries with a moving average"""
    movmean = lambda x: x.rolling(period, min_periods=period).mean()
    paths = [movmean(df) for df in paths]
    return paths


def plot_all_timeseries(series):
    for i, v in enumerate(series):
        info, files = v
        paths = load_simulation_paths(files.index.to_list())
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

    plt.rcParams['axes.prop_cycle'] = plt.cycler("color", plt.cm.tab10.colors)

    # Folders
    FOLDER_DEMAND = '/Users/karlnaumann/Desktop/Solow_Model/paths_demand/'
    FOLDER_GENERAL = '/Users/karlnaumann/Desktop/Solow_Model/paths_general/'
    T = 2000

    file_df = file_category(folder=FOLDER_GENERAL)
    series = file_df.groupby(['c2', 'gamma'])

    for i, v in enumerate(series):
        info, files = v
        info = pd.Series(info, index=['c2', 'gamma'])
        name = fig_name(FOLDER_GENERAL, info, 'histograms')
        business_cycle_hist_v2(info, files, T)

    # business_cycle_histograms(FOLDER_GENERAL, T, winsorize = True)

    # plot_all_timeseries(series)

    """"""

    """
    FOLDER = 'paths2'
    T = 250 # 3 months (3)

    file_df = file_category(folder=FOLDER)
    series = file_df.groupby(['c2', 'gamma'])

    fig, ax = plt.subplots(len(series), 1)

    for i, v in enumerate(series):
        info, files = v
        paths = load_simulation_paths(files.index.to_list())
        paths = [df.rolling(T, min_periods=T).mean() for df in paths]
        recessions = [cycle_timing(df.y, timescale=63) for df in paths]
        durations = pd.Series(cycle_durations(recessions, timescale=T))

        bounds = durations.quantile([0.05, 0.95])
        durations = durations[durations>bounds.min()]
        durations = durations[durations<bounds.max()]

        if len(series)>1:
            ax[i].hist(durations, bins=60)
            ax[i].set_xlabel('Duration (Years)')
            ax[i].set_title('c2 {:.1e} gamma {:.1e}'.format(*info))
        else:
            ax.hist(durations, bins=60)
            ax.set_xlabel('Duration (Years)')
            ax.set_title('c2 {:.1e} gamma {:.1e}'.format(*info))
    plt.tight_layout()
    plt.show()
    """
