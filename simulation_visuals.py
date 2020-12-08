import os
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as stat
from matplotlib import pyplot as plt
from matplotlib import rc

from demandSolow import DemandSolow
from phase_diagram import PhaseDiagram

# Graphing scripts

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


def heatmap(df: pd.DataFrame, label: str, save='', scatter=None, show=False, limits=None, freq=2):
    y_l = ['{:.1e}'.format(float(i)) for i in df.index]
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(7, 5)
    params = dict(
            cmap='coolwarm',
            # xticklabels = freq,
            # yticklabels = freq,
            linewidths=0.1,
            ax=ax,

    )

    col, ix = df.columns.to_list(), df.index.to_list()
    p2 = dict(xlim=(min(col), max(col)), ylim=(min(ix), max(ix)),
              xticks=sorted(col)[::freq], yticks=sorted(ix)[::freq])

    if limits is None:
        g = sns.heatmap(df.astype(float), cbar_kws={'label': label},
                        **params)

        # ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
        # ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        # ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 16)
        # ax.set_yticklabels(y_l[::2], fontsize = 16)
    else:
        g = sns.heatmap(df.astype(float), cbar_kws={'label': label},
                        **params, vmin=limits[0], vmax=limits[1])

    # ax.xaxis.set_major_formatter(ScalarFormatter())
    # ax.yaxis.set_major_formatter(ScalarFormatter())
    # ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax.set_xticks(np.arange(0, len(col), freq) + 0.5)
    ax.set_xticklabels(['{:.1f}'.format(i * 1e-3) for i in sorted(col)[::freq]])
    ax.set_yticks(np.arange(0, len(ix), freq) + 0.5)
    ax.set_yticklabels(['{:.1f}'.format(i * 1e4) for i in sorted(ix)[::freq]])
    ax.text(1.0, -0.05, '1e3', transform=ax.transAxes,
            horizontalalignment='right',
            verticalalignment='top')
    ax.text(-0.05, 1.0, '1e-4', transform=ax.transAxes,
            horizontalalignment='left',
            verticalalignment='bottom')
    # ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 16)
    # ax.set_yticklabels(y_l[::2], fontsize = 16)
    if isinstance(scatter, list):
        x, y = scatter
        ax.plot([i + 0.5 for i in x], [i + 0.5 for i in y], color='black',
                marker='x', linewidth=3.0)
    ax.tick_params(axis='x', labelrotation=0)
    ax.invert_yaxis()
    plt.tight_layout()
    if save != '':
        if '.png' not in save:
            save += '.png'
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    return ax


# Utility Scripts

def parse_filename(filename: str):
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
             'tau_h', 'c1', 'c2', 'beta1', 'beta2', 'gamma', 't_end']
    loc = [17, 18, 3, 15, 16, 12, 13, 14, 5, 7, 9, 11, 2, 1]
    cut = [4, 3, 1, 3, 3, 2, 2, 2, 0, 0, 0, 0, 1, 1]

    params = {p[0]: float(parts[p[1]][p[2]:]) for p in zip(names, loc, cut)}

    return params


def categorise_directory(folder: str = 'simulations', t_end='t1e+07'):
    """ Combine into dataframe, indexed by filename, the various parameters for
    the simulations that have been run

    Parameters
    ----------
    folder  :   str
    t_end   :   str

    Returns
    -------
    df  :   pd.DataFrame
    """
    files = [f for f in os.listdir(folder) if '.df' in f and t_end in f]
    data = {f: parse_filename(f) for f in files}
    df = pd.DataFrame(data).T
    df.index = [folder + '/' + i for i in df.index]
    return df, df.groupby(['beta1', 'beta2'])


def load_sims(files: list, t_end: str = 't1e+07'):
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



def analysis_dfs(sims: dict, cat_df: pd.DataFrame, epsilon: float = 1e-5):
    k = list(sims.keys())
    gamma_list = sorted(cat_df.loc[k, 'gamma'].unique().astype(float))
    c2_list = sorted(cat_df.loc[k, 'c2'].unique().astype(float))

    df_g = pd.DataFrame(index=c2_list, columns=gamma_list)
    df_div = pd.DataFrame(index=c2_list, columns=gamma_list)
    df_yks = pd.DataFrame(index=c2_list, columns=gamma_list)
    df_y_check = pd.DataFrame(index=c2_list, columns=gamma_list)
    df_check2 = pd.DataFrame(index=c2_list, columns=gamma_list)
    lim_g_min = pd.DataFrame(index=c2_list, columns=gamma_list)
    lim_g_max = pd.DataFrame(index=c2_list, columns=gamma_list)
    lim_div_min = pd.DataFrame(index=c2_list, columns=gamma_list)
    lim_div_max = pd.DataFrame(index=c2_list, columns=gamma_list)
    lim_yks_min = pd.DataFrame(index=c2_list, columns=gamma_list)
    lim_yks_max = pd.DataFrame(index=c2_list, columns=gamma_list)

    for g in gamma_list:
        for c2 in c2_list:
            # Check if relevant combo available
            key = cat_df[cat_df.loc[:, 'gamma'] == g]
            key = key[key.loc[:, 'c2'] == c2]
            if not key.index.empty:
                print(g, c2)
                # Results of parameter simulations
                df = sims[key.index[0]]
                df_g.loc[c2, g] = df.g.mean()
                print(df.g)
                df_div.loc[c2, g] = (df.psi_ks - df.psi_kd).mean()
                df_yks.loc[c2, g] = (df.psi_ks - df.psi_y).mean()
                df_y_check.loc[c2, g] = np.tanh(g * df.psi_y.mean())
                # Theoretical Check
                b2 = key.loc[key.index[0], 'beta2']
                df_check2.loc[c2, g] = c2 * g > 1 / (
                        b2 * df_g.loc[c2, g]) + epsilon
                # Empirical Check
                df_check2.loc[c2, g] = df_div.loc[c2, g] - epsilon < 0


                lim_g_min, lim_g_max = df_g.min(), df_g.max()
                lim_div_min = df_div.min()
                lim_div_max = df_div.max()
                lim_yks_min = df_yks.min()
                lim_yks_max = df_yks.max()

    df_check2.dropna(axis=1, inplace=True)
    df_g = df_g.loc[:, df_check2.columns]
    df_div = df_div.loc[:, df_check2.columns]
    df_yks = df_yks.loc[:, df_check2.columns]
    df_y_check = df_y_check.loc[:, df_check2.columns]

    limits = dict(g=(lim_g_min.min(), lim_g_max.max()),
                  divergence=(lim_div_min.min(), lim_div_max.max()),
                  yks=(lim_yks_min.min(), lim_yks_max.max()))

    for df in [df_g, df_div, df_yks]:
        df.index.name = 'c2'
        df.columns.name = 'Gamma'
        df = df.astype(float)
        df.index = [float(i) for i in df.index]

    return df_g, df_div, df_yks, df_y_check, df_check2, limits


def fig_name(b1: float, b2: float, kind: str):
    return 'fig_asymptotics_{}_b1_{:.1f}_b2_{:.1f}.png'.format(kind, b1, b2)


def divergence_loc(cutoff, df):
    q = df[df < cutoff].astype(float).idxmax(axis=1).dropna()
    g, c2 = list(df.columns), list(df.index)
    x = [g.index(i) for i in q]
    y = [c2.index(i) for i in q.index]
    return [x, y]


# Functions to Execute

def asymptotic_analyis(folder):
    cat_df, gb = categorise_directory()
    analysis = {}
    labels = ['g', 'Psi_ks-Psi_kd', 'Psi_ks-Psi_y']
    titles = ['g', 'divergence', 'yks']
    lims = [(0, 4), (-1.0e-6, 9e-6), (-1e-7, 3e-7)]
    b1, b2 = (1.0, 1.3), (0.7, 1.4)

    for b1b2, group in gb:
        if b1b2[0] > b1[0] and b1b2[0] < b1[1]:
            if b1b2[1] > b2[0] and b1b2[1] < b2[1]:
                sims = load_sims(group.index.to_list())
                dfs = analysis_dfs(sims, group)
                analysis[b1b2] = dict(g=dfs[0], div=dfs[1],
                                      yks=dfs[2], y_c=dfs[3])
                div_ix = divergence_loc(1e-8, dfs[1])
                mask = mask_check(dfs[4])
                freq = 2 if b1b2 != (1.1, 1.0) else 5
                for i, df in enumerate(dfs[:3]):
                    name = folder + fig_name(b1b2[0], b1b2[1], titles[i])
                    heatmap(df.mask(mask), labels[i], name, limits=lims[i],
                            freq=freq)

                if b1b2 == (1.1, 1.0):
                    for i, df in enumerate(dfs[:3]):
                        name = folder + fig_name(b1b2[0], b1b2[1], titles[i])
                        heatmap(df.mask(mask), labels[i],
                                name[:-4] + 'no_lim.png',
                                freq=freq)


def convergence_heatmap(folder, mask_epsilon):
    cat_df, gb = categorise_directory()
    analysis = {}
    labels = ['g', 'Psi_ks-Psi_kd', 'Psi_ks-Psi_y']
    titles = ['g', 'divergence', 'yks']
    lims = [(0, 4), (-1.0e-6, 9e-6), (-1e-7, 3e-7)]
    b1, b2 = (1.0, 1.3), (0.7, 1.4)

    for b1b2, group in gb:
        if b1b2 == (1.1, 1.0):
            sims = load_sims(group.index.to_list())
            dfs = analysis_dfs(sims, group, mask_epsilon)
            mask = mask_check(dfs[4])
            freq = 5
            for i, df in enumerate(dfs[:2]):
                name = folder + fig_name(b1b2[0], b1b2[1], titles[i])
                heatmap(df.mask(mask), labels[i], name[:-4] + 'no_lim.png',
                        freq=freq)


def mask_check(df):
    for j, r in enumerate(df.index):
        for i, c in enumerate(df.columns):
            if df.iloc[j, i]:
                df.iloc[j, i:] = True
                break
    return df.astype(bool)


def demand_limit_graphs(folder):
    params = dict(tech0=1, rho=1 / 3, epsilon=1e-5, tau_y=1000, tau_h=25,
                  tau_s=250, c1=1, c2=3.1e-4, gamma=2000, beta1=1.1, beta2=1.0,
                  s0=0.05)
    xi_args = dict(decay=0.2, diffusion=2.5)

    start = np.array([1, 0, 1, 0, 0, 0])
    start[0] = params['epsilon'] + params['rho'] * start[2]

    for gamma in [500, 1000, 2000, 4000]:
        for c2 in [1e-4, 2.5e-4, 4e-4]:
            for sig in [1.5, 2.5]:
                params['gamma'] = gamma
                params['c2'] = c2
                xi_args['diffusion'] = sig
                args = (gamma, c2, params['tau_y'], sig)
                name = 'fig_demand_g{:.0f}_c2_{:.1e}_tau{:.0f}_sig{:.1f}.png'.format(
                        *args)
                ds = DemandSolow(params, xi_args)
                ds.simulate(start, t_end=2e5, interval=0.1, seed=40, xi=True)
                ds.phase_diagram(save=folder + name)
                del ds


def demand_limit2(folder):
    p = dict(tau_h=25, tau_s=250, tau_y=1000, epsilon=1e-5, c1=1,
             c2=2.5e-4, s0=0, tech0=1, beta1=1.1, beta2=1.0,
             gamma=2000, phi=1.0, theta=0.2, sigma=2.5)

    start = np.array([1, 0, 1, 0])
    start[0] = p['epsilon'] + start[2] / 3

    pd = PhaseDiagram(**p)
    pd.overview(start, plot=True, t_end=1e5)

    for gamma in [500, 2000, 4000]:
        for c2 in [1e-4, 2.5e-4, 3e-4]:
            for sig in [1.5, 2.5]:
                p['gamma'], p['c2'], p['sigma'] = gamma, c2, sig
                args = (gamma, c2, p['tau_y'], sig)
                name = 'fig_demand_g{:.0f}_c2_{:.1e}_tau{:.0f}_sig{:.1f}.png'.format(
                        *args)
                pd = PhaseDiagram(**p)
                pd.overview(start, plot=True, t_end=1e5, save=folder + name)


if __name__ == '__main__':
    folder = '/Users/karlnaumann/Library/Mobile Documents/com~apple~CloudDocs/Econophysics/Project_SolowModel/Paper/figures/'
    folder = '/Users/karlnaumann/Library/Mobile Documents/com~apple~CloudDocs/Econophysics/Project_SolowModel/Test_figures/'

    mask_epsilon = -2e-8
    convergence_heatmap(folder, mask_epsilon)

    # g_analysis(folder)

    # asymptotic_analyis(folder)
    # supply_limit_graph(folder)
    # demand_limit_graphs(folder)
    # demand_limit2(folder)
