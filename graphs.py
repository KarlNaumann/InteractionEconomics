import os
import pickle

import numpy as np
import pandas as pd
import utilities as ut

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy import stats as ss

# UTILITIES

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


def load_paths(file_list, period):
    paths = [v for i, v in ut.load_sims(file_list).items()]
    return smoothing(paths, period)


def _winsorize(series, perc=[0.05, 0.95]):
    quint = series.quantile(perc)
    new = series[series > quint.min()]
    return new[new < quint.max()]


def smoothing(paths: list, period: int):
    """Smooth the timeseries with a moving average across p periods"""
    return [df.rolling(period, min_periods=period).mean() for df in paths]


# ANALYSIS

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


# CLASSIC SOLOW FUNCTIONS
def boundary_layer_approximation(t_end: float, b: float, eps: float, rho: float, tau_y: float, lam: float, dep: float):
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


def classic_solow_growth_path(t_end: float, start: list, eps: float, rho: float, tau_y: float, lam: float, dep: float):
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
        path[t, 0] = path[t - 1, 0] + v_y / tau_y
        path[t, 1] = path[t - 1, 1] + v_k
    return path


def supply_limit_plot(params: dict=dict(rho=1 / 3, eps=1e-5, tau_y=1e3, lam=0.2, dep=0.02), const:float=1.5, t_end=1e5, save: str=''):

    bla = boundary_layer_approximation(t_end, const, **params)

    # Starting values are in real terms and match the BLA
    ln_y0 = np.log(bla[0])
    ln_k0 = ln_y0 / params['rho']

    solow = classic_solow_growth_path(t_end, [ln_y0, ln_k0], **params)

    data = np.hstack([bla[:, np.newaxis], np.exp(solow[:, 0, np.newaxis])])
    data = pd.DataFrame(data, columns=['Boundary Layer Approx.', 'Solow'])

    fig = plt.figure(figsize=(ut.page_width(), ut.page_width()/2))
    ax = fig.add_subplot()

    labels = dict(ytxt='Production', xtxt='Time')
    ut.time_series_plot(data, ax, **labels)

    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.legend(ncol=len(data.columns), loc=4)

    # Inset axis to highlight the adjustment period
    axins = ax.inset_axes([0.1, 0.5, 0.47, 0.47])
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec='0.8', linestyle='--')
    ut.time_series_plot(data.iloc[int(2e3):int(2e4), :], axins)
    axins.get_legend().remove()

    fig.tight_layout()

    if save != '':
        plt.savefig(save, bbox_inches='tight', format='eps')
    else:
        plt.show()


# PLOTTING HELPERS

def histogram(x, bins: int=20, win:bool=True, save='', xtxt='Duration (Years)', show=False, info=False):

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(ut.page_width(), 3)
    
    n, bins = np.histogram(x, bins=bins)
    n = n / x.shape[0]

    ax.hist(bins[:-1], bins, weights=n, edgecolor='black')
    ax.set_xlabel(xtxt)
    ax.set_ylabel('Proportion')

    if info:
        textstr = '\n'.join([
            r'mean ${:.2f}$'.format(np.mean(x)),
            r'median ${:.2f}$'.format(np.median(x)),
            r'std. ${:.2f}$'.format(np.std(x))])
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.8, 0.8, textstr, transform=ax.transAxes, 
                verticalalignment='top', bbox=props, fontsize=10)


    plt.tight_layout()
    if save != '':
        plt.savefig(save, bbox_inches='tight', format='eps')
    if show:
        plt.show()
    else:
        plt.show()


def formatter(value, tick_num):
        return value / 250


def timseries_plot(df, info, stop=int(1e6), save=''):
    # General figure of a path
    fig, axs = plt.subplots(3,1)

    # Production
    axs[0].plot(df.y.iloc[:stop])
    axs[0].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    axs[0].xaxis.major.formatter._useMathText = True
    axs[0].set_xlabel(r'Time $t$ (business days)')
    axs[0].set_ylabel(r'Log Production $y$')

    # Capital Timeseries
    axs[1].plot(df.ks.iloc[:stop], label=r'Supply')
    axs[1].plot(df.kd.iloc[:stop], label=r'Demand')
    axs[1].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    axs[1].xaxis.major.formatter._useMathText = True
    axs[1].legend()
    axs[1].set_xlabel(r'Time $t$ (business days)')
    axs[1].set_ylabel(r'Log Capital')

    # Sentiment timeseries
    axs[2].plot(df.s.iloc[:stop])
    axs[2].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    axs[2].xaxis.major.formatter._useMathText = True
    axs[2].set_xlabel(r'Time $t$ (business days)')
    axs[2].set_ylabel(r'Sentiment $s$')

    fig.align_ylabels()
    plt.tight_layout()

    if save != '':
        plt.savefig(save, bbox_inches='tight', format='eps')
        plt.close()
    else:
        fig.suptitle('c2 {:.1e} gamma {:.1e}'.format(*info))
        plt.show()


def total_hist(dfs, info, stop=int(1e6), save='', show=False):

    ys = [df.y.diff()[df.ks >= df.kd] for df in dfs]
    y = np.vstack([x.values[:, np.newaxis] for x in ys])
    ks = [df.loc[:, ['ks', 'kd']].min(axis=1).diff()[df.ks >= df.kd] for df in dfs]
    k = np.vstack([x.values[:, np.newaxis] for x in ks])
    s = np.vstack([x.s.values[:, np.newaxis] for x in dfs])

    # General figure of a path
    fig, axs = plt.subplots(1,3)
    fig.set_size_inches(ut.page_width(),ut.page_width()/3)

    # Mini-histogram of growth rates
    axs[0].hist(y, bins=50, edgecolor='black')
    axs[0].set_xlim(-2e-4, 2e-4)
    axs[0].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    axs[0].xaxis.major.formatter._useMathText = True
    axs[0].get_xaxis().get_offset_text().set_visible(False)
    #axs[0].xaxis.offsetText.set_fontsize(8)
    axs[0].set_yticks([], minor=True)
    axs[0].set_yticks([])
    axs[0].set_xlabel(r'Growth $y$ for $k_d<k_s$ $(\times10^{-4})$')

    # Mini-histogram of growth rates
    axs[1].hist(k, bins=50, edgecolor='black')
    axs[1].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    axs[1].xaxis.major.formatter._useMathText = True
    axs[1].get_xaxis().get_offset_text().set_visible(False)
    #axs[1].xaxis.offsetText.set_fontsize(8)
    axs[1].set_yticks([], minor=True)
    axs[1].set_yticks([])
    axs[1].set_xlabel(r'Growth $k$ for $k_d<k_s$ $(\times10^{-4})$')

    # Sentiment histogram
    axs[2].hist(s, bins=50, edgecolor='black')
    axs[2].set_yticks([], minor=True)
    axs[2].set_yticks([])
    axs[2].set_xlabel(r'Sentiment $s$')

    plt.tight_layout()

    if save != '':
        plt.savefig(save, bbox_inches='tight', format='eps')
    if show:
        plt.show()
    else:
        plt.close()


def fig_name(folder, info, kind):
    struct = [
        'g_{:.0f}_'.format(info.gamma),
        'c2_{:.0f}_'.format(info.c2 * 1e5),
        'tauh_{:.0f}_'.format(info.tau_h),
        'taus_{:.0f}_'.format(info.tau_s),
        'tauy_{:.0f}_'.format(info.tau_y)
    ]

    try:
        struct.append('seed_{:.0f}'.format(info.seed))
    except AttributeError:
        pass

    return '{}fig_{}_{}'.format(folder, kind, ''.join(struct))


# FUNCTIONS FOR THE USER

def cycle_duration_histograms(series, var:list, period:int=2000, bins:int=20, winsorize=False, folder='', show=False, g_info=True):
    """Histogram of business cycle durations per gamnma c2 combination"""
    for i, v in enumerate(series):
        info, files = v
        info = pd.Series(info, index=var)
        name = fig_name(folder, info, 'histograms')
        _, df = cycle_analysis(load_paths(files.index, period))
        x = _winsorize(df.duration, perc=(0.00,.95)) if winsorize else df.duration
        histogram(x / 250, save=name+'.eps', bins=bins, show=show, info=g_info)

def cycle_duration_histogram_mini(series, var:list, period:int=2000, n_bins:int=20, winsorize=False, folder='', show=False, g_info=True):
    """Histogram of business cycle durations per gamnma c2 combination"""
    from matplotlib import gridspec
    for i, v in enumerate(series):
        info, files = v
        info = pd.Series(info, index=var)
        save = fig_name(folder, info, 'histograms')
        save=''
        _, df = cycle_analysis(load_paths(files.index, period))
        v = df.values
        dur_exp = pd.Series(v[:, 1] - v[:, 0])
        dur_rec = pd.Series(v[1:, 0] - v[:-1, 1])

        fig = plt.figure()
        gs = gridspec.GridSpec(2, 3)
        fig.set_size_inches(ut.page_width(), 3)
        
        # Main Business Cycle Duration Graph
        ax_main = fig.add_subplot(gs[:, :2])
        x = _winsorize(df.duration, perc=(0.00,.95)) if winsorize else df.duration
        n, bins = np.histogram(x / 250, bins=n_bins)
        n = n / x.shape[0]
        ax_main.hist(bins[:-1], bins, weights=n, edgecolor='black')
        ax_main.set_xlabel('Duration (Years)')
        ax_main.set_ylabel('Proportion')
        
        ax_exp = fig.add_subplot(gs[1, 2])
        x = _winsorize(dur_exp[dur_exp>0], perc=(0.00, .95)) if winsorize else dur_exp[dur_exp>0]
        n, bins = np.histogram(x / 250, bins=n_bins)
        n = n / x.shape[0]
        ax_exp.hist(bins[:-1], bins, weights=n, edgecolor='black')
        ax_exp.set_xlabel('Expansion Duration (Years)')

        ax_rec = fig.add_subplot(gs[0, 2])
        x = _winsorize(dur_rec[dur_rec>0], perc=(0.00, .95)) if winsorize else dur_rec[dur_rec>0]
        n, bins = np.histogram(x / 250, bins=n_bins)
        n = n / x.shape[0]
        ax_rec.hist(bins[:-1], bins, weights=n, edgecolor='black')
        ax_rec.set_xlabel('Recession Duration (Years)')

        plt.tight_layout()
        if save != '':
            plt.savefig(save, bbox_inches='tight', format='eps')
        if show:
            plt.show()
        else:
            plt.show()



def capital_ratio_histograms(series, var:list, period:int=2000, bins:int=20, folder='', show=False, g_info=True):
    """Histogram of the capital supply to demand ratio per gamnma c2 combination"""
    for i, v in enumerate(series):
        info, files = v
        info = pd.Series(info, index=var)
        name = fig_name(folder, info, 'ratio_hist')
        paths = load_paths(files.index, period)
        df = np.vstack([(p.kd/p.ks).dropna().values[:, np.newaxis] for p in paths])
        histogram(df, save=name+'.eps', bins=bins, xtxt=r'$k_d/k_s$', show=show, info=g_info)


def plot_all_timeseries(series, vars:list, period:int=2000, folder:str=''):
    """ Timeseries plot of all simulations including overall histograms"""
    for i, v in enumerate(series):
        info, files = v
        info = pd.Series(info, index=vars)
        paths = load_paths(files.index, int(period))

        for i, df in enumerate(paths):
            info.loc['seed'] = i
            save = fig_name(folder, info, 'timeseries')
            timseries_plot(df, info, stop=int(4e5), save=save+'.eps')


def plot_bimodal_histograms(series, vars:list, period:int=2000, folder:str='', show=False):
    """ Plot histograms of sentiment, production growth and capital growth """
    for i, v in enumerate(series):
        info, files = v
        info = pd.Series(info, index=vars)
        paths = load_paths(files.index, int(period))
        save = fig_name(folder, info, 'timeseries_hist')
        total_hist(paths, info, save = save+'.eps', show=show)



def plot_test1(series, vars:list, period:int=2000, folder:str=''):
    """ Plot histograms of sentiment, production growth and capital growth """
    for i, v in enumerate(series):
        info, files = v
        info = pd.Series(info, index=vars)
        paths = load_paths(files.index, int(period))
        save = fig_name(folder, info, 'timeseries_hist')
        
        # General figure of a path
        fig, axs = plt.subplots(1,2)
        fig.set_size_inches(10,5)

        # Strength of Terms in Dynamical Equations
        axs[0].set_xlabel(r'Strength of terms in $h$')

        ys = [4000*df.y.diff() for df in paths]
        y = np.vstack([x.values[:, np.newaxis] for x in ys])
        axs[0].hist(y, bins=50, edgecolor='black', color='white', label=r'$\gamma\dot{y}$')
        
        xis = [df.xi for df in paths]
        xi = np.vstack([x.values[:, np.newaxis] for x in xis])
        axs[0].hist(xi, bins=50, edgecolor='orange', color='orange', alpha=0.5, label=r'$\xi_t$')

        axs[0].legend()

        # Mini-histogram of growth rates
        axs[1].set_xlabel(r'Strength of terms in $s$')

        b1s = [1.1*df.s for df in paths]
        b1s = np.vstack([x.values[:, np.newaxis] for x in b1s])
        axs[1].hist(b1s, bins=50, edgecolor='black', color='white', label=r'$\beta_1s$')

        
        b2h = [1.0*df.h for df in paths]
        b2h = np.vstack([x.values[:, np.newaxis] for x in b2h])
        axs[1].hist(b2h, bins=50, edgecolor='green', alpha=0.5, color='red', label=r'$h$')
        
        b2h = [2.0*df.h for df in paths]
        b2h = np.vstack([x.values[:, np.newaxis] for x in b2h])
        axs[1].hist(b2h, bins=50, edgecolor='orange', alpha=0.5, color='orange', label=r'$2h$')
        
        axs[1].legend()
        
        plt.tight_layout()
        save=''
        if save != '':
            plt.savefig(save, bbox_inches='tight', format='png')
            plt.close()
        else:
            fig.suptitle('c2 {:.1e} gamma {:.1e}'.format(*info))
            plt.show()


def plot_test2(series, vars:list, period:int=1e5, folder:str='', n:int=5):
    """ Timeseries plot of all simulations including overall histograms"""
    for i, v in enumerate(series):
        info, files = v
        info = pd.Series(info, index=vars)
        paths = load_paths(files.index, int(2000))

        fig, axs = plt.subplots(n,2)
        fig.set_size_inches(10,10)
        
        for i, df in enumerate(paths[:n]):
            period = df.shape[0]
            # Sentiment timeseries
            axs[i,0].plot(df.s.iloc[:period])
            axs[i,0].xaxis.set_major_formatter(formatter)
            #axs[i,0].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
            axs[i,0].xaxis.major.formatter._useMathText = True
            axs[i,0].set_xlabel(r'Time $t$ (business days)')
            axs[i,0].set_ylabel(r'Sentiment $s$')

            # Capital Timeseries
            axs[i,1].plot(df.ks.iloc[:period], label=r'$k_s$')
            axs[i,1].plot(df.kd.iloc[:period], label=r'$k_d$')
            axs[i,1].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
            axs[i,1].xaxis.major.formatter._useMathText = True
            axs[i,1].legend()
            axs[i,1].set_xlabel(r'Time $t$ (business days)')
            axs[i,1].set_ylabel(r'Log Capital')

        plt.tight_layout()

        save = folder + 'fig_simulation_dynamics.png'
        
        if folder != '':
            plt.savefig(save, bbox_inches='tight', format='png')
            plt.close()
        else:
            plt.show()
        

def test3(series, var:list, period:int=2000, bins:int=20, winsorize=False, folder='', show=False, g_info=True):
    """Histogram of business cycle durations per gamnma c2 combination"""
    for i, v in enumerate(series):
        info, files = v
        info = pd.Series(info, index=var)
        name = fig_name(folder, info, 'histograms')
        _, df = cycle_analysis(load_paths(files.index, period))
        x = _winsorize(df.duration, perc=(0.00,.95)) if winsorize else df.duration
        histogram(x / 250, save='', bins=bins, show=show, info=g_info)

if __name__ == '__main__':

    ut.plot_settings()

    # Folders
    PATH_FIGS = 'figures2/'
    PATH_ASYM = 'data_simulations_asymptotics/'
    PATH_DEM = 'data_simulations_paths_demand/'
    PATH_GEN = 'data_simulations_paths_general_sig2.0/'
    PATH_EMP = 'data_empirical/'
    T = 2000

    file_df = ut.parse_directory(folder=PATH_GEN)

    for i, v in enumerate(file_df.groupby(['c2', 'gamma'])):
        info, files = v
        info = pd.Series(info, index=['c2', 'gamma'])
        name = fig_name(FOLDER_GENERAL, info, 'histograms')
        business_cycle_histograms(info, files, T)
