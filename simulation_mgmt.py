import os
import pickle
import sys
import time
import tqdm
from itertools import product
from multiprocessing import cpu_count, get_context

import numpy as np
import pandas as pd

from solowModel import SolowModel
from demandSolow import DemandSolow


# HELPER FUNCTIONS
def name_gen(p, t_end, folder='computations/', seed=None) -> str:
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
    if seed is not None:
        return folder + name + '_seed_{:d}'.format(seed) + '.df'
    else:
        return folder + name + '.df'


def extract_info(filename, kind):
    parts = filename.split('_')
    t = np.float(parts[1][1:])
    seed, gamma, c2 = None, None, None
    for i, part in enumerate(parts):
        if part[0] == 'g' and part[1] != 'e':
            gamma = int(part[1:])
        if part == 'c2':
            c2 = float(parts[i + 1])
        if 'seed' in part:
            seed = int(parts[i + 1][:-3])
    if kind == 'path':
        return t, gamma, c2, seed
    else:
        return t, gamma, c2


def init_val():
    # Starting value
    start = np.array([1, 10, 9, 0, 0, 1, 0])
    start[0] = 1e-5 + (min(start[1:3]) / 3)
    return start


# MULTIPROCESSING FUNCTIONS
def worker_path(args):
    # Initialise
    sm = SolowModel(params=args[0], xi_args=args[1])
    seeds, t_end, start, folder = args[2:]

    for i, seed in enumerate(seeds):
        # Simulate
        path = sm.simulate(start, t_end=t_end, seed=seed)
        # Save
        file = open(name_gen(args[0], t_end, folder=folder, seed=seed), 'wb')
        pickle.dump(path, file)
        # Close
        file.close()
    del sm


def worker_asymp(args):
    # Initialise
    sm = SolowModel(params=args[0], xi_args=args[1])
    seeds, t_end, start, folder = args[2:]
    # Simulate
    df = pd.DataFrame(index=seeds,
                      columns=['psi_y', 'psi_ks', 'psi_kd', 'g', 'sbar_hat',
                               'sbar_theory', 'sbar_crit'])
    for i, seed in enumerate(seeds):
        sm.simulate(start, t_end=t_end, seed=seed)
        df.loc[seed, :] = sm.asymptotics()
    # Save
    file = open(name_gen(args[0], t_end, folder=folder), 'wb')
    pickle.dump(df, file)
    # Close
    file.close()
    del sm
    del df


def worker_demand_limit(args):
    # Initialise
    sm = DemandSolow(params=args[0], xi_args=args[1])
    seeds, t_end, start, folder = args[2:]

    for i, seed in enumerate(seeds):
        # Simulate
        path = sm.simulate(start, t_end=t_end, seed=seed)
        if i == 0:
            sm.phase_diagram()
        # Save
        p = dict(args[0])
        p['saving0'], p['dep'], p['h_h'] = 0,0,0
        file = open(name_gen(p, t_end, folder=folder, seed=seed), 'wb')
        pickle.dump(path, file)
        # Close
        file.close()
    del sm


def pool_mgmt(worker, tasks):
    n_tasks, t = len(tasks), time.time()
    arg = (n_tasks, cpu_count(), time.strftime("%H:%M:%S", time.gmtime(t)))
    print("Starting\t{} processes on {} CPUs at {}".format(*arg))

    with get_context("spawn").Pool() as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(worker, tasks), total=len(tasks), position=0, leave=True):
            pass


# SIMULATION HELPERS
def task_creator(variations: dict, folder: str, seeds: list, t_end: float, start: np.ndarray, kind: str = 'asymptotic', xi_args: dict = dict(decay=0.2, diffusion=2.0)):
    # Set up the simulation batch parameters
    params = dict(tech0=1, rho=1 / 3, epsilon=1e-5, tau_y=1000, dep=0.0002,
                  tau_h=25, tau_s=250, c1=1, c2=3.1e-4, gamma=2000, beta1=1.1,
                  beta2=1.0, saving0=0.15, h_h=10)

    # Set up the task list generation
    files = [f for f in os.listdir(folder) if '.df' in f]
    tasks = []
    var_p = list(variations.keys())

    if kind == 'path':
        exist = [extract_info(f, kind) for f in files]
        # Iterate through all combinations
        for tup in product(*list(variations.values())):
            p = dict(params)
            # Update parameter dictionary
            for combo in zip(var_p, tup):
                p[combo[0]] = combo[1]
            # Check if sim already happened or not
            for seed in seeds:
                if name_gen(p, t_end, folder='', seed=seed) not in files:
                    arg = (p, xi_args, seeds, t_end, start, folder)
                    tasks.append(arg)

    elif kind == 'asymptotic':
        exist = [extract_info(f, kind) for f in files]
        # Iterate through all combinations
        for tup in product(*list(variations.values())):
            p = dict(params)
            # Update parameter dictionary
            for combo in zip(var_p, tup):
                p[combo[0]] = combo[1]

            arg = (p, xi_args, seeds, t_end, start, folder)
            if name_gen(p, t_end, folder='') not in files:
                tasks.append(arg)

    elif kind == 'demand_limit':
        exist = [extract_info(f, 'path') for f in files]
        params2 = dict(tech0=1, rho=1 / 3, epsilon=1e-5, tau_y=1000,
                  tau_h=25, tau_s=250, c1=1, c2=3.1e-4, gamma=2000, beta1=1.1,
                  beta2=1.0)
        start = np.array([1, 0, 9, 0, 0, 0])
        start[0] = np.exp(start[1])
        # Iterate through all combinations
        for tup in product(*list(variations.values())):
            p = dict(params2)
            p2 = dict(params)
            # Update parameter dictionary
            for combo in zip(var_p, tup):
                p[combo[0]] = combo[1]
                p2[combo[0]] = combo[1]

            arg = (p, xi_args, seeds, t_end, start, folder)
            if name_gen(p2, t_end, folder='') not in files:
                tasks.append(arg)

    return tasks


# OPTIONS
def case_b1_paths():
    # Interesting Variations
    b1_variations = dict(
            gamma=np.arange(1000, 4000, 500),
            c2=np.arange(1e-4, 5e-4, 5e-5),
            beta1=[1.1, 1.2, 1.3])

    # Set up all of the combinations to use
    tasks = task_creator(b1_variations, folder='simulations/', seeds=[1],
                         t_end=1e7, start=init_val(), kind='path')

    # Run the multiprocessed simulations
    pool_mgmt(worker_path, tasks)


def case_b2_paths():
    # Interesting Variations
    b2_variations = dict(
            gamma=np.arange(1000, 4000, 500),
            c2=np.arange(1e-4, 5e-4, 5e-5),
            beta2=[1.1, 1.2, 1.3])

    # Set up all of the combinations to use
    tasks = task_creator(b2_variations, folder='simulations/', seeds=[1],
                         t_end=1e7, start=init_val(), kind='path')

    # Run the multiprocessed simulations
    pool_mgmt(worker_path, tasks)


def case_c1_paths():
    # Interesting Variations
    c1_variations = dict(
            gamma=np.arange(1000, 4000, 500),
            c2=np.arange(1e-4, 5e-4, 5e-5),
            c1=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
    )

    # Set up all of the combinations to use
    tasks = task_creator(c1_variations, folder='simulations/', seeds=[1],
                         t_end=1e7, start=init_val(), kind='path')

    # Run the multiprocessed simulations
    pool_mgmt(worker_path, tasks)


def case_b1_asymp():
    # Interesting Variations
    b1_variations = dict(
            gamma=np.arange(1000, 4000, 250),
            c2=np.arange(1e-4, 5e-4, 2e-5),
            beta1=[1.1, 1.2, 1.3])

    # Set up all of the combinations to use
    seeds = list(range(5))
    tasks = task_creator(b1_variations, folder='simulations/', seeds=seeds,
                         t_end=1e7, start=init_val(), kind='asymptotic')

    # Run the multiprocessed simulations
    pool_mgmt(worker_asymp, tasks)


def case_b2_asymp():
    # Interesting Variations
    b2_variations = dict(
            gamma=np.arange(1000, 4000, 250),
            c2=np.arange(1e-4, 5e-4, 2e-5),
            beta2=[0.5, 0.6, 0.7, 0.8, 0.9])

    # Set up all of the combinations to use
    seeds = list(range(5))
    tasks = task_creator(b2_variations, folder='simulations/', seeds=seeds,
                         t_end=1e7, start=init_val(), kind='asymptotic')

    # Run the multiprocessed simulations
    pool_mgmt(worker_asymp, tasks)


def case_finegrain_asymp():
    # Interesting Variations
    variations = dict(
            gamma=np.arange(1000, 6000, 100),
            c2=np.arange(1e-4, 6e-4, 1e-5))

    # Set up all of the combinations to use
    seeds = list(range(25))
    tasks = task_creator(variations, folder='simulations/', seeds=seeds,
                         t_end=1e7, start=init_val(), kind='asymptotic')

    # Run the multiprocessed simulations
    pool_mgmt(worker_asymp, tasks)


def case_paths():
    # Interesting Variations
    b1_variations = dict(
            gamma= [6000],
            c2 = [3e-4])

    seeds = list(range(1,11))

    # Set up all of the combinations to use
    tasks = task_creator(b1_variations, folder='paths/', seeds=seeds,
                         t_end=1e6, start=init_val(), kind='path',
                         xi_args = dict(decay=0.2, diffusion=1.0))

    # Run the multiprocessed simulations
    pool_mgmt(worker_path, tasks)


def case_path_test():
    # Interesting Variations
    variations = dict(gamma= [4000], c2 = [2.5e-4])

    seeds = list(range(1,31))
    folder = '/Users/karlnaumann/Desktop/Solow_Model/paths_general/'

    # Set up all of the combinations to use
    tasks = task_creator(variations, folder=folder, seeds=seeds,
                         t_end=1e6, start=init_val(), kind='path',
                         xi_args = dict(decay=0.2, diffusion=2.5))

    # Run the multiprocessed simulations
    pool_mgmt(worker_path, tasks)


def case_demand_test():
    # Interesting Variations
    variations = dict(gamma= [6000], c2 = [2.5e-4])

    seeds = list(range(1,11))#[54]
    folder = '/Users/karlnaumann/Desktop/Solow_Model/paths_demand/'


    # Set up all of the combinations to use
    tasks = task_creator(variations, folder=folder, seeds=seeds,
                         t_end=1e6, start=init_val(), kind='demand_limit',
                         xi_args = dict(decay=0.2, diffusion=2.5))

    # Run the multiprocessed simulations
    for args in tasks:
        worker_demand_limit(args)


def demand_limit_gamma_variations():
    """Generate the phase diagrams for variations in the gamma parameter"""

    variations = [1000, 4000, 7000]
    seeds = [42]
    folder = '/Users/karlnaumann/Desktop/Solow_Model/paths_demand/'
    params = dict(tech0=1, rho=1 / 3, epsilon=1e-5, tau_y=1000, tau_h=25,
                  tau_s=250, c1=1, c2=2.5e-4, gamma=2000, beta1=1.1, beta2=1.0,
                  s0=-0.05)

    start = np.array([1, 0, 9, 0, 0, 0])
    start[0] = np.exp(start[1])

    for gamma in variations:
        params['gamma'] = gamma
        sm = DemandSolow(params=params, xi_args=dict(decay=0.2, diffusion=2.0))
        for seed in seeds:
            sm.simulate(start, t_end=1e6, seed=seed)
            info = pd.Series([params['c2'], params['gamma']],
                             index=['c2', 'gamma'])
            sm.phase_diagram(save=fig_name(folder, info, 'limit_phase_diagram'))


def demand_limit_c2_variations():
    """Generate the phase diagrams for variations in the gamma parameter"""
    variations = [1.5e-4, 2.5e-4, 3.5e-4]
    seeds = [42]
    folder = '/Users/karlnaumann/Desktop/Solow_Model/paths_demand/'
    params = dict(tech0=1, rho=1 / 3, epsilon=1e-5, tau_y=1000, tau_h=25,
                  tau_s=250, c1=1, c2=2.5e-4, gamma=4000, beta1=1.1,
                  beta2=1.0, s0=-0.05)

    start = np.array([1, 0, 9, 0, 0, 0])
    start[0] = np.exp(start[1])

    for c2 in variations:
        params['c2'] = c2
        sm = DemandSolow(params=params,
                         xi_args=dict(decay=0.2, diffusion=2.0))
        for seed in seeds:
            sm.simulate(start, t_end=1e6, seed=seed)
            info = pd.Series([params['c2'], params['gamma']],
                             index=['c2', 'gamma'])
            sm.phase_diagram(
                save=fig_name(folder, info, 'limit_phase_diagram'))


def fig_name(folder, info, kind):
    struct = ['g_{:.0f}_'.format(info.gamma),
              'c2_{:.0f}_'.format(info.c2 * 1e5)]

    try:
        struct.append('seed_{:.0f}'.format(info.seed))
    except AttributeError:
        pass

    return '{}fig_{}_{}.png'.format(folder, kind, ''.join(struct))

if __name__ == '__main__':
    globals()[sys.argv[1]]()






