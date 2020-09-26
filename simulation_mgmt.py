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
        return folder + name + 'seed_{:d}'.format(seed) + '.df'
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


# MULTIPROCESSING FUNCTIONS

def worker_path(args):
    # Initialise
    sm = SolowModel(params=args[0], xi_args=args[1])
    seed, t_end, start, folder = args[2:]
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


def pool_mgmt(worker, tasks):
    n_tasks = len(tasks)
    t = time.time()
    arg = (n_tasks, cpu_count(), time.strftime("%H:%M:%S", time.gmtime(t)))
    print("Starting\t{} processes on {} CPUs at {}".format(*arg))

    with get_context("spawn").Pool() as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(worker, tasks), total=len(tasks), position=0, leave=True):
            pass
    """
    with get_context("spawn").Pool() as pool:
        for i, _ in enumerate(pool.imap_unordered(worker, tasks), 1):
            sys.stderr.write('\rCompleted: {0:.2%}'.format(i / n_tasks))
        pool.close()
        pool.join()
    """
    print('\n Total Time: {}'.format(
            time.strftime("%H:%M:%S", time.gmtime(time.time() - t))))


# SIMULATION HELPERS

def task_creator(variations: dict, folder: str, seeds: list, t_end: float,
                 start: np.ndarray, kind: str = 'asymptotic',
                 xi_args: dict = dict(decay=0.2, diffusion=2.0)):
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

    return tasks


# OPTIONS

def init_val():
    # Starting value
    start = np.array([1, 10, 9, 0, 0, 1, 0])
    start[0] = 1e-5 + (min(start[1:3]) / 3)
    return start


def case_b1_paths():
    # Interesting Variations
    b1_variations = dict(
            gamma=np.arange(1000, 4000, 100),
            c2=np.arange(1e-4, 5e-4, 2e-5),
            beta1=[1.1, 1.2, 1.3])

    # Set up all of the combinations to use
    tasks = task_creator(b1_variations, folder='simulations/', seeds=[1],
                         t_end=1e7, start=init_val(), kind='path')

    # Run the multiprocessed simulations
    pool_mgmt(worker_path, tasks)


def case_b2_paths():
    # Interesting Variations
    b2_variations = dict(
            gamma=np.arange(1000, 4000, 250),
            c2=np.arange(1e-4, 5e-4, 2e-5),
            beta2=[1.1, 1.2, 1.3])

    # Set up all of the combinations to use
    tasks = task_creator(b2_variations, folder='simulations/', seeds=[1],
                         t_end=1e7, start=init_val(), kind='path')

    # Run the multiprocessed simulations
    pool_mgmt(worker_path, tasks)


def case_c1_paths():
    # Interesting Variations
    c1_variations = dict(
            gamma=np.arange(1000, 4000, 100),
            c2=np.arange(1e-4, 5e-4, 2e-5),
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


if __name__ == '__main__':
    globals()[sys.argv[1]]()
