import pickle

import numpy as np
import pandas as pd
from solowModel_cython import SolowModel


def name_gen(p, t_end, folder: str = 'asymptotic_simulations/') -> str:
    parts = [
        'general',
        't{:05.0e}'.format(t_end),
        'g{:05.0f}'.format(p['gamma']),
        'e{:07.1e}'.format(p['epsilon']),
        'c1_{:03.1f}'.format(p['c1']),
        'c2_{:07.1e}'.format(p['c2']),
        'b1_{:03.1f}'.format(p['beta1']),
        'b2_{:03.1f}'.format(p['beta2']),
        'ty{:03.0f}'.format(p['tau_y']),
        'ts{:03.0f}'.format(p['tau_s']),
        'th{:02.0f}'.format(p['tau_h']),
        'lam{:01.2f}'.format(p['saving0']),
        'dep{:07.1e}'.format(p['dep']),
        'tech{:04.2f}'.format(p['tech0']),
        'rho{:04.2f}'.format(p['rho']),
    ]

    name = '_'.join(parts)
    name = folder + name + '.df'
    return name


# Parameters for the simulation
params = dict(tech0=1, rho=1 / 3, epsilon=1e-5, tau_y=1000, dep=0.0002,
              tau_h=25, tau_s=250, c1=1, c2=3.1e-4, gamma=2000, beta1=1.1,
              beta2=1.0, saving0=0.15, h_h=10)
xi_args = dict(decay=0.2, diffusion=2.0)
start = np.array([1, 10, 9, 0, 0, 1, 0])
start[0] = params['epsilon'] + params['rho'] * min(start[1:3])
t_end = 1e7
sm = SolowModel(params, xi_args)

# Parameters to investigate
gamma_list = [1000, 1500, 2000, 2500, 3000]
c2_list = np.linspace(1e-4, 4e-4, 11)
seeds = list(range(100))

cols = ['psi_y', 'psi_ks', 'psi_kd', 'g', 'sbar_hat', 'sbar_theory',
        'sbar_crit']

for g, gamma in enumerate(gamma_list):
    print('## Gamma = {} ({}/{})'.format(gamma, g, len(gamma_list))
    sm.params['gamma'] = gamma
    for c, c2 in enumerate(c2_list):
        print('## C2 = {} ({}/{})'.format(c2, c, len(c2_list))
        sm.params['c2'] = c2
        df = pd.DataFrame(index=seeds, columns=cols)
        for i, seed in enumerate(seeds):
            sm.simulate(start, t_end=t_end, seed=seed)
            df.loc[seed, :] = sm.asymptotics()

        file = open(name_gen(params, t_end), 'wb')
        pickle.dump(df, file)
        file.close()
        print("Done")
