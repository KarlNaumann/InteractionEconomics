import os
import pickle
import sys
import time
from itertools import product
from multiprocessing import cpu_count, get_context

import numpy as np
from solowModel_cython import SolowModel


# HELPER FUNCTIONS

def name_gen(p, t_end, folder: str = 'computations/', seed=None) -> str:
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


def extract_info(filename):
    parts = filename.split('_')
    t = np.float(parts[1][1:])
    seed, gamma, c2 = None, None, None
    for i, part in enumerate(parts):
        if part[0] == 'g' and part[1] != 'e':
            gamma = int(part[1:])
        if part == 'c2':
            c2 = float(parts[i + 1])
        if 'seed' in part:
            seed = int(parts[i + 1])
    if seed is not None:
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
    name = name_gen(sm.params, t_end, folder=folder, seed=seed)
    file = open(name, 'wb')
    pickle.dump(path, file)
    # Close
    file.close()
    del sm


def pool_mgmt(worker, tasks):
    n_tasks = len(tasks)
    with get_context("spawn").Pool() as pool:
        for i, _ in enumerate(pool.imap_unordered(worker, tasks), 1):
            sys.stderr.write('\rCompleted: {0:.2%}'.format(i / n_tasks))
        pool.close()
        pool.join()


# SIMULATION HELPERS

def task_creator(variations: dict, folder: str, seeds: list, duration: float,
                 start: np.ndarray,
                 xi_args: dict = dict(decay=0.2, diffusion=2.0)):
    # Set up the simulation batch parameters
    params = dict(tech0=1, rho=1 / 3, epsilon=1e-5, tau_y=1000, dep=0.0002,
                  tau_h=25, tau_s=250, c1=1, c2=3.1e-4, gamma=2000, beta1=1.1,
                  beta2=1.0, saving0=0.15, h_h=10)

    # Check for completed parameter combinations
    files = [f for f in os.listdir(folder) if '.df' in f]
    done_sets = [extract_info(f) for f in files]

    tasks = []
    var_p = list(variations.keys())
    # Iterate through all combinations
    for tup in product(*list(variations.values())):
        # Update parameter dictionary
        for combo in zip(var_p, tup):
            params[combo[0]] = combo[1]
        # Check if sim already happened or not
        for seed in seeds:
            if (duration, params['gamma'], params['c2'], seed) not in done_sets:
                arg = (params, xi_args, seed, duration, start, folder)
                tasks.append(arg)
    return tasks


if __name__ == '__main__':

    # Starting value
    start = np.array([1, 10, 9, 0, 0, 1, 0])
    start[0] = 1e-5 + (min(start[1:3]) / 3)

    # Interesting Variations
    b1_variations = dict(
            gamma=np.arange(1000, 4000, 100),
            c2=np.arange(1e-4, 5e-4, 2e-5),
            beta1=[1.1, 1.2, 1.3],
    )

    # Set up all of the combinations to use
    tasks = task_creator(b1_variations, folder='simulations/', seeds=[1],
                         duration=1e7, start=start)

    # Run the multiprocessed simulations
    t = time.time()
    arg = (len(tasks), cpu_count(), time.strftime("%H:%M:%S", time.gmtime(t)))
    print("Starting B1 Variations\t{} processes on {} CPUs at {}".format(*arg))
    pool_mgmt(worker_path, tasks)
    print('\n Total Time: {}'.format(
        time.strftime("%H:%M:%S", time.gmtime(time.time() - t))))

    # Interesting Variations
    b2_variations = dict(
            gamma=np.arange(1000, 4000, 100),
            c2=np.arange(1e-4, 5e-4, 2e-5),
            beta2=[1.1, 1.2, 1.3],
    )

    # Set up all of the combinations to use
    tasks = task_creator(b1_variations, folder='simulations/', seeds=[1],
                         duration=1e7, start=start)

    # Run the multiprocessed simulations
    t = time.time()
    arg = (len(tasks), cpu_count(), time.strftime("%H:%M:%S", time.gmtime(t)))
    print("Starting B2 Variations\t{} processes on {} CPUs at {}".format(*arg))
    pool_mgmt(worker_path, tasks)
    print('\n Total Time: {}'.format(
            time.strftime("%H:%M:%S", time.gmtime(time.time() - t))))

    # Interesting Variations
    c1_variations = dict(
            gamma=np.arange(1000, 4000, 100),
            c2=np.arange(1e-4, 5e-4, 2e-5),
            c1=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
    )

    # Set up all of the combinations to use
    tasks = task_creator(b1_variations, folder='simulations/', seeds=[1],
                         duration=1e7, start=start)

    # Run the multiprocessed simulations
    t = time.time()
    arg = (len(tasks), cpu_count(), time.strftime("%H:%M:%S", time.gmtime(t)))
    print("Starting C1 Variations\t{} processes on {} CPUs at {}".format(*arg))
    pool_mgmt(worker_path, tasks)
    print('\n Total Time: {}'.format(
            time.strftime("%H:%M:%S", time.gmtime(time.time() - t))))