import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from solowModel import SolowModel
from classicSolow import ClassicSolow

FIG_PATH = '/Users/karlnaumann/Library/Mobile Documents/com~apple~CloudDocs/Econophysics/Internship_SolowModel/Draft_Paper/figures/'

# Graphing scripts

def heatmap(df:pd.DataFrame, label:str, save='', scatter=None,
            show=False):
    y_l = ['{:.1e}'.format(float(i)) for i in df.index]
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(12,8)
    sns.heatmap(df.astype(float), cmap='coolwarm', cbar_kws={'label': label},
                yticklabels=y_l, xticklabels=2, linewidths=.1, ax=ax)
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

    for g in gammas:
        for c2 in c2s:
            key = cat_df[cat_df.loc[:,'gamma']==g][cat_df.loc[:,'c2']==c2]
            df = sims[key.index[0]]
            df_g.loc[c2,g]=df.g.mean()
            df_div.loc[c2,g]=(df.psi_ks - df.psi_kd).mean()
            df_yks.loc[c2,g]=(df.psi_ks - df.psi_y).mean()
            df_y_check.loc[c2,g] = np.tanh(g*df.psi_y.mean())

    for df in [df_g, df_div, df_yks]:
        df.index.name='c2'
        df.columns.name='Gamma'
        df = df.astype(float)
        df.index = [float(i) for i in df.index]

    return df_g, df_div, df_yks, df_y_check


def fig_name(b1:float, b2:float, kind:str):
    return 'fig_asymptotics_{}_b1_{:.1f}_b2_{:.1f}.png'.format(kind, b1, b2)

def divergence_loc(cutoff, df):
    q = df[df<cutoff].astype(float).idxmax(axis=1).dropna()
    g, c2 = list(df.columns), list(df.index)
    x = [g.index(i) for i in q]
    y = [c2.index(i) for i in q.index]
    return [x, y]

# Functions to Execute

def supply_limit_graph():
    cs = ClassicSolow(params)
    cs.simulate([20, 3], 1e5)
    cs.visualise(save=FIG_PATH+'fig_limitks.png')

def asymptotic_analyis():
    cat_df, gb = file_category()
    analysis = {}
    labels = ['g', 'Psi_ks-Psi_kd', 'Psi_ks-Psi_y']
    titles = ['g', 'divergence', 'yks']

    for b1b2, group in gb:
        sims = load_sims(group.index.to_list())
        dfs = analysis_dfs(sims, group)
        analysis[b1b2] = dict(g=dfs[0], div=dfs[1],
                              yks=dfs[2], y_c=dfs[3])
        div_ix =  divergence_loc(1e-6, dfs[1])
        for i, df in enumerate(dfs[:3]):
            name = FIG_PATH+fig_name(b1b2[0], b1b2[1], titles[i])
            heatmap(df, labels[i], name, div_ix)

if __name__=='__main__':
    asymptotic_analyis()

