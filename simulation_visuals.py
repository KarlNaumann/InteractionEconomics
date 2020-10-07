import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as stat

from matplotlib import pyplot as plt
from solowModel import SolowModel
from classicSolow import ClassicSolow
from demandSolow import DemandSolow
from phase_diagram import PhaseDiagram

font = {'family' : 'normal',
        'size'   : 12}

plt.rc('font', **font)

# Graphing scripts

def heatmap(df:pd.DataFrame, label:str, save='', scatter=None, show=False, limits=None):
    y_l = ['{:.1e}'.format(float(i)) for i in df.index]
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(7,5)
    params = dict(
        cmap='coolwarm',
        xticklabels = 2,
        yticklabels = 2,
        linewidths = 0.1,
        ax = ax,

    )
    if limits is None:
        sns.heatmap(df.astype(float), cbar_kws={'label': label},
                        **params)
        #ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 16)
        #ax.set_yticklabels(y_l[::2], fontsize = 16)
    else:
        g = sns.heatmap(df.astype(float), cbar_kws={'label': label},
            **params, vmin=limits[0], vmax=limits[1])
        #ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 16)
        #ax.set_yticklabels(y_l[::2], fontsize = 16)
    if isinstance(scatter, list):
        x,y = scatter
        ax.plot([i+0.5 for i in x], [i+0.5 for i in y], color='black',
                marker='x', linewidth=3.0)
    ax.tick_params(axis='x', labelrotation= 0)
    ax.invert_yaxis()
    plt.tight_layout()
    if save != '':
        if '.png' not in save:
            save +='.png'
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    return ax


# Utility Scripts

def name_extract(filename:str)->dict:
    parts = filename[:-3].split('_')
    params = {}
    variables = ['tech0','rho','epsilon','saving0','dep',
            'tau_y','tau_s','tau_h','c1','c2',
            'beta1', 'beta2', 'gamma']
    locs = [17, 18, 3, 15, 16, 12, 13, 14, 5, 7, 9, 11, 2]
    cuts = [4, 3, 1, 3, 3, 2, 2, 2, 0, 0, 0, 0, 1]

    for p in zip(variables, locs, cuts):
        params[p[0]] = float(parts[p[1]][p[2]:])

    t_end = float(parts[1][1:])

    return params, t_end


def file_category(folder:str='simulations', t_end:str='t1e+07'):
    files = [f for f in os.listdir(folder) if '.df' in f and t_end in f]
    p,_ = name_extract(files[0])
    df = pd.DataFrame(index = list(range(len(files))),
                      columns=list(p.keys()) + ['t_end', 'file'])
    for i, f in enumerate(files):
        p, t_end = name_extract(f)
        df.iloc[i, :-2] = p
        df.loc[i, 't_end'] = t_end
        df.loc[i, 'file'] = folder+'/'+f

    df.set_index('file', inplace=True)
    return df, df.groupby(['beta1','beta2'])


def load_sims(files:list, t_end:str='t1e+07'):
    sims = {}
    for path in files:
        file = open(path, 'rb')
        df = pickle.load(file)
        sims[path] = df
        file.close()
    return sims


def analysis_dfs(sims:dict, cat_df:pd.DataFrame):

    k = list(sims.keys())
    gammas = sorted(cat_df.loc[k,'gamma'].unique())
    c2s = sorted(cat_df.loc[k,'c2'].unique())

    df_g = pd.DataFrame(index=c2s, columns=gammas)
    df_div = pd.DataFrame(index=c2s, columns=gammas)
    df_yks = pd.DataFrame(index=c2s, columns=gammas)
    df_y_check = pd.DataFrame(index=c2s, columns=gammas)
    df_check2 = pd.DataFrame(index=c2s, columns=gammas)
    lim_g_min = pd.DataFrame(index=c2s, columns=gammas)
    lim_g_max = pd.DataFrame(index=c2s, columns=gammas)
    lim_div_min = pd.DataFrame(index=c2s, columns=gammas)
    lim_div_max = pd.DataFrame(index=c2s, columns=gammas)
    lim_yks_min = pd.DataFrame(index=c2s, columns=gammas)
    lim_yks_max = pd.DataFrame(index=c2s, columns=gammas)

    for g in gammas:
        for c2 in c2s:
            key = cat_df[cat_df.loc[:,'gamma']==g]
            key = key[key.loc[:,'c2']==c2]
            df = sims[key.index[0]]
            df_g.loc[c2,g]=df.g.mean()
            df_div.loc[c2,g]=(df.psi_ks - df.psi_kd).mean()
            df_yks.loc[c2,g]=(df.psi_ks - df.psi_y).mean()
            df_y_check.loc[c2,g] = np.tanh(g*df.psi_y.mean())
            b2 = key.loc[key.index[0],'beta2']
            df_check2.loc[c2, g] = c2*g > 1 / (b2*df_g.loc[c2, g])
            lim_g_min, lim_g_max = df_g.min(), df_g.max()
            lim_div_min = df_div.min()
            lim_div_max = df_div.max()
            lim_yks_min = df_yks.min()
            lim_yks_max = df_yks.max()

    limits = dict(g=(lim_g_min.min(), lim_g_max.max()),
                  divergence=(lim_div_min.min(), lim_div_max.max()),
                  yks=(lim_yks_min.min(), lim_yks_max.max()))

    for df in [df_g, df_div, df_yks]:
        df.index.name='c2'
        df.columns.name='Gamma'
        df = df.astype(float)
        df.index = [float(i) for i in df.index]

    return df_g, df_div, df_yks, df_y_check, df_check2, limits


def fig_name(b1:float, b2:float, kind:str):
    return 'fig_asymptotics_{}_b1_{:.1f}_b2_{:.1f}.png'.format(kind, b1, b2)

def divergence_loc(cutoff, df):
    q = df[df<cutoff].astype(float).idxmax(axis=1).dropna()
    g, c2 = list(df.columns), list(df.index)
    x = [g.index(i) for i in q]
    y = [c2.index(i) for i in q.index]
    return [x, y]

# Functions to Execute

def supply_limit_graph(folder):
    params = dict(tech0=1, rho=1 / 3, epsilon=1e-5, tau_y=1000, dep=0.0002,
                  tau_h=25, tau_s=250, c1=1, c2=3.1e-4, gamma=2000, beta1=1.1,
                  beta2=1.0, saving0=0.15, h_h=10)
    cs = ClassicSolow(params)
    cs.simulate([20, 3], 1e5)
    cs.visualise(save=folder+'fig_limitks.png')

def asymptotic_analyis(folder):
    cat_df, gb = file_category()
    analysis = {}
    labels = ['g', 'Psi_ks-Psi_kd', 'Psi_ks-Psi_y']
    titles = ['g', 'divergence', 'yks']
    lims = [(0, 4),(-1.0e-6, 9e-6),(-1e-7, 3e-7)]
    b1, b2 = (1.0, 1.3), (0.7, 1.4)

    for b1b2, group in gb:
        if b1b2[0]>b1[0] and b1b2[0]<b1[1]:
            if b1b2[1]>b2[0] and b1b2[1]<b2[1]:
                sims = load_sims(group.index.to_list())
                dfs = analysis_dfs(sims, group)
                analysis[b1b2] = dict(g=dfs[0], div=dfs[1],
                                       yks=dfs[2], y_c=dfs[3])
                div_ix =  divergence_loc(1e-6, dfs[1])
                for i, df in enumerate(dfs[:3]):
                    name = folder+fig_name(b1b2[0], b1b2[1], titles[i])
                    heatmap(df.mask(dfs[4]), labels[i], name, div_ix,
                        limits = lims[i])

def demand_limit_graphs(folder):
    params = dict(tech0=1, rho=1 / 3, epsilon=1e-5, tau_y=1000,tau_h=25,
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
                name = 'fig_demand_g{:.0f}_c2_{:.1e}_tau{:.0f}_sig{:.1f}.png'.format(*args)
                ds = DemandSolow(params, xi_args)
                ds.simulate(start, t_end=2e5, interval=0.1, seed=40, xi=True)
                ds.phase_diagram(save=folder+name)
                del ds

def demand_limit2(folder):

    p = dict(tau_h = 25, tau_s = 250, tau_y = 1000, epsilon = 1e-5, c1 = 1,
             c2 = 2.5e-4, s0 = 0, tech0 = 1, beta1 = 1.1, beta2 = 1.0,
             gamma = 2000, phi = 1.0, theta = 0.2, sigma = 2.5)

    start = np.array([1, 0, 1, 0])
    start[0] = p['epsilon'] + start[2]/3

    pd = PhaseDiagram(**p)
    pd.overview(start, plot=True, t_end=1e5)


    for gamma in [500, 2000, 4000]:
        for c2 in [1e-4, 2.5e-4, 3e-4]:
            for sig in [1.5, 2.5]:
                p['gamma'], p['c2'], p['sigma'] = gamma, c2, sig
                args = (gamma, c2, p['tau_y'], sig)
                name = 'fig_demand_g{:.0f}_c2_{:.1e}_tau{:.0f}_sig{:.1f}.png'.format(*args)
                pd = PhaseDiagram(**p)
                pd.overview(start, plot=True, t_end=1e5, save=folder+name)

def g_analysis(folder):
    cat_df, gb = file_category()
    for b1b2, group in gb:
        if b1b2==(1.1, 1.0):
            sims = load_sims(group.index.to_list())

    ix = list(range(len(list(sims.keys()))))
    y = pd.Series(index=ix, name='g')
    x = pd.DataFrame(index=ix, columns=['gamma', 'c2'])

    for i,k in enumerate(sims.keys()):
        y.iloc[i] = sims[k].g.mean()
        x.loc[i, :] = cat_df.loc[k, x.columns.to_list()]

    x.loc[:,'g_c2'] = x.gamma * x.c2
    x.loc[:,'ln_g'] = np.log(x.astype(float).gamma)
    x.loc[:,'ln_c2'] = np.log(x.astype(float).c2)
    x.loc[:,'ln_prod'] = x.ln_g * x.ln_c2
    x.loc[:,'const'] = 1

    model = stat.OLS(y.astype(float),x.iloc[:,-3:].astype(float))
    results = model.fit()
    print(x.columns)
    print(results.summary())
    fig, ax = plt.subplots(2,2)
    ax[0,0].scatter(y, results.fittedvalues)
    ax[0,1].scatter(x.c2, results.resid)
    ax[1,0].plot(results.resid)
    ax[1,1].hist(results.resid, bins = 20)
    plt.show()
    plt.plot(results.resid)



if __name__=='__main__':
    folder = '/Users/karlnaumann/Library/Mobile Documents/com~apple~CloudDocs/Econophysics/Internship_SolowModel/Draft_Paper/figures/'
    folder = '/Users/karlnaumann/Library/Mobile Documents/com~apple~CloudDocs/Econophysics/Internship_SolowModel/Test_figures/'


    g_analysis(folder)

    #asymptotic_analyis(folder)
    #supply_limit_graph(folder)
    #demand_limit_graphs(folder)
    #demand_limit2(folder)



