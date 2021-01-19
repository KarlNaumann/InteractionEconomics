import os
import numpy as np
import pandas as pd
import utilities as ut

from demandSolow import DemandSolow
from phase_diagram import PhaseDiagram


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
                # Results of parameter simulations
                df = sims[key.index[0]]
                df_g.loc[c2, g] = df.g.mean()
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


# Functions to Execute

def asymptotic_analyis(data_folder, save_folder):
    cat_df = ut.parse_directory(data_folder, criteria=['.df'])
    gb = cat_df.groupby(['beta1', 'beta2'])
    analysis = {}
    labels = ['g', 'Psi_ks-Psi_kd', 'Psi_ks-Psi_y']
    titles = ['g', 'divergence', 'yks']
    lims = [(0, 4), (-1.0e-6, 9e-6), (-1e-7, 3e-7)]
    b1, b2 = (1.0, 1.3), (0.7, 1.4)

    for b1b2, group in gb:
        if b1b2[0] > b1[0] and b1b2[0] < b1[1]:
            if b1b2[1] > b2[0] and b1b2[1] < b2[1]:
                sims = ut.load_sims(group.index.to_list())
                dfs = analysis_dfs(sims, group)
                analysis[b1b2] = dict(g=dfs[0], div=dfs[1],
                                      yks=dfs[2], y_c=dfs[3])
                mask = mask_check(dfs[4])
                freq = 2 if b1b2 != (1.1, 1.0) else 5
                for i, df in enumerate(dfs[:3]):
                    name = save_folder + fig_name(b1b2[0], b1b2[1], titles[i])
                    ut.c2_gamma_heatmap(df.mask(mask), labels[i], name,
                                        limits=lims[i], freq=freq)

                if b1b2 == (1.1, 1.0):
                    for i, df in enumerate(dfs[:3]):
                        name = save_folder + fig_name(b1b2[0], b1b2[1], titles[i])
                        ut.c2_gamma_heatmap(df.mask(mask), labels[i],
                                            name[:-4] + 'no_lim.png', freq=freq)


def convergence_heatmap(data_folder, save_folder):
    cat_df = ut.parse_directory(data_folder, criteria=['.df'])
    gb = cat_df.groupby(['beta1', 'beta2'])
    labels = ['g', r'$\Psi_{ks}-\Psi_{kd}$', r'$\Psi_{ks}-\Psi_{y}$']
    titles = ['g', 'divergence', 'yks']
    for b1b2, group in gb:
        if b1b2 == (1.1, 1.0):
            sims = ut.load_sims(group.index.to_list())
            dfs = analysis_dfs(sims, group, 0)
            mask = mask_check(dfs[4])
            for i, df in enumerate(dfs[:2]):
                name = save_folder + fig_name(b1b2[0], b1b2[1], titles[i])
                ut.c2_gamma_heatmap(df.mask(mask), labels[i],
                                    name[:-4] + 'no_lim.eps', freq=5)


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
                name = 'fig_demand_g{:.0f}_c2_{:.1e}_tau{:.0f}_sig{:.1f}.eps'.format(
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
                name = 'fig_demand_g{:.0f}_c2_{:.1e}_tau{:.0f}_sig{:.1f}.png'. format(*args)
                pd = PhaseDiagram(**p)
                pd.overview(start, plot=True, t_end=1e5, save=folder + name)


if __name__ == '__main__':

    ut.plot_settings()

    folder = '/'.join(os.getcwd().split('/')[:-1] + ['figures/'])
    # folder = '/'.join(os.getcwd().split('/')[:-1] + ['Test_figures/'])

    mask_epsilon = 0
    convergence_heatmap('simulations/', folder)

    # g_analysis(folder)

    # asymptotic_analyis(folder)
    # supply_limit_graph(folder)
    # demand_limit_graphs(folder)
    # demand_limit2(folder)
