# Test file

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from classicSolow import ClassicSolow
from demandSolow import DemandSolow
from solowModel import SolowModel

def def_params(demand=False):
    if demand:
        return dict(tech0=1, rho=1 / 3, epsilon=1e-5, tau_y=1000,
                    tau_h=25, tau_s=250, c1=1, c2=2.5e-4, gamma=1000, beta1=1.1,
                    beta2=1.0)

    return dict(tech0=1, rho=1 / 3, epsilon=1e-5, tau_y=1000, dep=0.0002,
              tau_h=25, tau_s=250, c1=1, c2=2.5e-4, gamma=2000, beta1=1.1,
              beta2=1.0, saving0=0.15, h_h=10)

# Initialise parameters
params = def_params()

xi_args = dict(decay=0.2, diffusion=1.5)
start = np.array([1, 10, 9, 0, 0, 1, params['saving0']])
start[0] = params['epsilon'] + params['rho'] * min(start[1:3])

sm = SolowModel(params=params, xi_args=xi_args)

general = False
supply_limit = False
demand_limit = False
dimitri_doublecheck = False
phase_examples = False
timescale_borders = False
saving_rate = False
fig_sh_phases = True

folder = '/Users/karlnaumann/Library/Mobile Documents/com~apple~CloudDocs/Econophysics/Internship_SolowModel/Test_figures/'

if general:
    # General case
    df = sm.simulate(start, t_end=1e7, seed=1)
    sm.visualise(save=folder+'fig_test_run_1e7')
    df = sm.simulate(start, t_end=1e5, seed=1)
    sm.visualise(save=folder+'fig_test_run_1e5')

if supply_limit:
    cs = ClassicSolow(params)
    cs.simulate([20, 3], 1e5)
    cs.visualise(save='figures/fig_limitks')

if demand_limit:
    params = dict(tech0=1, rho=1 / 3, epsilon=1e-5, tau_y=1000,
              tau_h=25, tau_s=250, c1=1, c2=2.5e-4, gamma=1000, beta1=1.1,
              beta2=1.0)
    ds = DemandSolow(params, xi_args)
    start = np.array([1, 0, 1, 0, 0, 0])
    start[0] = params['epsilon'] + params['rho'] * start[2]
    ds.simulate(start, t_end=2e5, interval=0.1, seed=40, xi=True)
    ds.phase_diagram()

if dimitri_doublecheck:
    params = dict(tech0=1, rho=1 / 3, epsilon=1e-5, tau_y=2000, dep=0.0002,
              tau_h=25, tau_s=250, c1=1, c2=2.5e-4, gamma=2000, beta1=1.1,
              beta2=1.0, saving0=0.15, h_h=10)

    xi_args = dict(decay=0.2, diffusion=2.5)
    start = np.array([1, 10, 9, 0, 0, 1, params['saving0']])
    start[0] = params['epsilon'] + params['rho'] * min(start[1:3])

    gc2 = [(0.0001, 11000), (0.00005, 11000), (0.00008, 13750),
           (0.00009, 12222.2), (0.00007, 15714.3)]


    for i, v in enumerate(gc2):
        params['gamma'] = v[1]
        params['c2'] = v[0]

        sm = SolowModel(params=params, xi_args=xi_args)
        df = sm.simulate(start, t_end=1e7, seed=40)
        a = sm.asymptotics()
        info = '-----------\n'
        info += 'c2 {:.3e}, gamma {:.0f} \n'.format(*v)
        info += 'g {:.3e}\n '.format(a[3])
        info += 'Psi_y {:.2e}\t Psi_y {:.2e}'.format(a[0], a[2])
        print(info)
        del sm

if phase_examples:
    params_A1 = dict(tech0=1, rho=1 / 3, epsilon=1e-5, tau_y=1000, dep=0.0002,
              tau_h=25, tau_s=250, c1=1, c2=4e-4, gamma=3000, beta1=1.1,
              beta2=1.0, saving0=0.15, h_h=10)
    params_A2 = dict(tech0=1, rho=1 / 3, epsilon=1e-5, tau_y=1000, dep=0.0002,
              tau_h=25, tau_s=250, c1=1, c2=1e-4, gamma=1000, beta1=1.1,
              beta2=1.0, saving0=0.15, h_h=10)
    params_B = dict(tech0=1, rho=1 / 3, epsilon=1e-5, tau_y=1000, dep=0.0002,
              tau_h=25, tau_s=250, c1=3, c2=4e-4, gamma=3000, beta1=1.1,
              beta2=1.0, saving0=0.15, h_h=10)
    params_C = dict(tech0=1, rho=1 / 3, epsilon=1e-5, tau_y=1000, dep=0.0002,
              tau_h=25, tau_s=250, c1=1, c2=2e-4, gamma=30000, beta1=1.1,
              beta2=1.0, saving0=0.15, h_h=10)


    xi_args = dict(decay=0.2, diffusion=1.5)
    start = np.array([1, 10, 9, 0, 0, 1, params['saving0']])
    start[0] = params['epsilon'] + params['rho'] * min(start[1:3])

    q = [('A1', params_A1, 'fig_phase_A1_g3000_c2_4e-4'),
         ('A2', params_A2, 'fig_phase_A2_g1000_c2_1e-4'),
         ('B', params_B, 'fig_phase_B_g1000_c2_8e-4'),
         ('C', params_C, 'fig_phase_C_g11000_c2_2e-4')]

    for v in q:
        sm = SolowModel(params=v[1], xi_args=xi_args)
        df = sm.simulate(start, t_end=1e7, seed=40)
        sm.visualise(save=folder+v[2]+'_t1e7')
        df = sm.simulate(start, t_end=1e5, seed=40)
        sm.visualise(save=folder+v[2]+'_t1e5')

if timescale_borders:
    params_ty1 = dict(tech0=1, rho=1 / 3, epsilon=1e-5, tau_y=1, dep=0.0002,
              tau_h=25, tau_s=250, c1=1, c2=4e-4, gamma=3000, beta1=1.1,
              beta2=1.0, saving0=0.15, h_h=10)
    params_ts1 = dict(tech0=1, rho=1 / 3, epsilon=1e-5, tau_y=1000, dep=0.0002,
              tau_h=25, tau_s=1, c1=1, c2=4e-4, gamma=3000, beta1=1.1,
              beta2=1.0, saving0=0.15, h_h=10)
    params_th1 = dict(tech0=1, rho=1 / 3, epsilon=1e-5, tau_y=1000, dep=0.0002,
              tau_h=1, tau_s=250, c1=1, c2=4e-4, gamma=3000, beta1=1.1,
              beta2=1.0, saving0=0.15, h_h=10)
    params_t1 = dict(tech0=1, rho=1 / 3, epsilon=1e-5, tau_y=1, dep=0.0002,
              tau_h=1, tau_s=1, c1=1, c2=4e-4, gamma=3000, beta1=1.1,
              beta2=1.0, saving0=0.15, h_h=10)
    params_ty250 = dict(tech0=1, rho=1 / 3, epsilon=1e-5, tau_y=250,
              dep=0.0002, tau_h=25, tau_s=250, c1=1, c2=4e-4, gamma=3000,
              beta1=1.1, beta2=1.0, saving0=0.15, h_h=10)



    xi_args = dict(decay=0.2, diffusion=1.5)
    start = np.array([1, 10, 9, 0, 0, 1, params['saving0']])
    start[0] = params['epsilon'] + params['rho'] * min(start[1:3])

    q = [('A1', params_ty1, 'fig_ty1_g3000_c2_4e-4'),
         ('A2', params_ts1, 'fig_ts1_g3000_c2_4e-4'),
         ('B', params_th1, 'fig_th1_g3000_c2_4e-4'),
         ('C', params_t1, 'fig_t1_g3000_c2_4e-4'),
         ('C', params_ty250, 'fig_ty250_g3000_c2_4e-4')]

    for v in q[-1:]:
        sm = SolowModel(params=v[1], xi_args=xi_args)
        df = sm.simulate(start, t_end=1e7, seed=40)
        sm.visualise(save=folder+v[2]+'_t1e7')
        df = sm.simulate(start, t_end=1e5, seed=40)
        sm.visualise(save=folder+v[2]+'_t1e5')

if saving_rate:
    df_gdp = pd.read_csv('data/GDPC1.csv')
    df_gdp.set_index(['DATE'], inplace=True)
    df_inv = pd.read_csv('data/GPDIC1.csv')
    df_inv.set_index(['DATE'], inplace=True)
    df = df_inv.merge(df_gdp,how='inner', left_index=True, right_index=True)
    print(df.head())
    df.loc[:,'saving'] = df.GPDIC1 / df.GDPC1
    fig, ax = plt.subplots(1,1)
    df.saving.plot()
    ax.axhline(df.saving.mean())
    plt.show()

    df_a = pd.read_csv('data/assets.csv')
    df_a.set_index(['DATE'], inplace=True)
    df_d = pd.read_csv('data/dep.csv')
    df_d.set_index(['DATE'], inplace=True)
    df = df_a.merge(df_d, how='inner', left_index=True, right_index=True)
    df.columns = ['a','d']
    print(df.head())
    df.loc[:,'dep_rate_ann'] = df.d / df.a
    fig, ax = plt.subplots(1,1)
    df.dep_rate_ann.plot()
    ax.axhline(df.dep_rate_ann.mean())
    plt.show()

if fig_sh_phases:
    params_A1 = def_params()
    xi_args = dict(decay=0.2, diffusion=1.5)
    start = np.array([1, 10, 9, 0, 0, 1, params['saving0']])
    start[0] = params['epsilon'] + params['rho'] * min(start[1:3])

    q = []
    for g in [1000,2000,3000]:
        for c2 in [2e-4, 3e-4, 4e-4]:
            params = def_params(demand=True)
            params['gamma'] = g
            params['c2'] = c2
            name = "fig_sh_phase_g{:.0f}_c2_{:.0e}".format(g, c2)
            ds = DemandSolow(params, xi_args)
            ds.sh_phase(save=folder+name)





