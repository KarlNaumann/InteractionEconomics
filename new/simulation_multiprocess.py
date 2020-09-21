import os
import pickle
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from solowModel_cython import SolowModel


def name_gen(p, t_end, folder: str = 'test/') -> str:
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

def extract_g_c2(filename):
    parts = filename.split('_')
    # locate g
    for i,part in enumerate(parts):
        if part[0] == 'g':
            gamma = int(part[1:])
        if part == 'c2':
            c2 = float(parts[i+1])

    return (gamma, c2)


def sim_worker(args):
    sm.params['gamma'], sm.params['c2'], seeds, t_end = args
    print("Sim: gamma={}, c2={}".format(args[0],args[1]))
    df = pd.DataFrame(index=seeds,
                      columns=['psi_y', 'psi_ks', 'psi_kd', 'g', 'sbar_hat',
                               'sbar_theory', 'sbar_crit'])
    for i, seed in enumerate(seeds):
        sm.simulate(start, t_end=t_end, seed=seed)
        df.loc[seed, :] = sm.asymptotics()

    file = open(name_gen(params, t_end, folder='asymptotic_simulations'), 'wb')
    pickle.dump(df, file)
    file.close()


if __name__ == '__main__':

    params = dict(tech0=1, rho=1 / 3, epsilon=1e-5, tau_y=1000, dep=0.0002,
                  tau_h=25, tau_s=250, c1=1, c2=3.1e-4, gamma=2000, beta1=1.1,
                  beta2=1.0, saving0=0.15, h_h=10)

    xi_args = dict(decay=0.2, diffusion=2.0)
    start = np.array([1, 10, 10, 0, 0, 1, params['saving0']])
    start[0] = params['epsilon'] + params['rho'] * min(start[1:3])

    sm = SolowModel(params=params, xi_args=xi_args)

    duration = 1e6
    # Parameters to investigate
    gamma_list = np.arange(1000, 4100, 100)  # [1000, 1500, 2000, 2500, 3000]
    c2_list = np.arange(1e-4, 5e-4, 2e-5)  # np.linspace(1e-4, 4e-4, 11)
    seed_list = list(range(100))

    # Completed sets
    files = [f for f in os.listdir() if 'general' in f]
    done_pairs = [extract_g_c2(f) for f in files]

    work_log = []
    for gamma in gamma_list:
        for c2 in c2_list:
            if (gamma, float("{:.6f}".format(c2))) not in done_pairs:
                work_log.append((gamma, c2, seed_list, duration))

    t = time.time()
    time_now = time.strftime("%H:%M:%S", time.gmtime(t))
    print("Starting {} processes on {} CPUs at {}".format(len(work_log),
                                                          cpu_count(), time_now))

    p = Pool(processes=cpu_count())
    p.map(sim_worker, tuple(work_log))

    time_tot = time.strftime("%H:%M:%S", time.gmtime(time.time()-t))
    print("Completed {} processes on {} CPUs in {}".format(len(work_log),
                                                          cpu_count(),
                                                          time_now))
