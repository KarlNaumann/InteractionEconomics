import os
import pickle

import numpy as np
import pandas as pd

from solowModel import SolowModel


def sim_models(parameters: dict, initial_values: np.ndarray,
               xi_args=None,
               t_end: float = 1e6,
               seeds=None,
               case: str = 'general',
               save_loc: str = 'pickles/',
               verbose: bool = False) -> None:
    """ Simulate a series of 20 dynamic Solow Models given a dictionary of
    parameters

    Parameters
    ----------
    parameters  :   dict
    initial_values   :   np.ndarray
    xi_args :   dict (optional)
    t_end   :   float
    seeds   :   list
    case    :   str
    save_loc:   str
    verbose :   bool

    Returns
    -------

    """

    if seeds is None:
        seeds = list(range(20))

    if xi_args is None:
        xi_args = dict(decay=0.2, diffusion=1.0)

    sm = SolowModel(parameters, xi_args=xi_args)

    for i, seed in enumerate(seeds):
        if verbose:
            print("Seed {} ({}/{})".format(seed, i, len(seeds)))
        _ = sm.solve(initial_values, t_end, seed=seed, case=case, save=False)
        sm.save_model(folder=save_loc)

    return None


def directory_dataframe(save_loc: str = 'pickles/'):
    file_list = os.listdir(save_loc)

    cols = ['t_end', 'gamma', 'epsilon', 'c1', 'c2', 'beta1', 'beta2', 'tau_y',
            'tau_s',
            'tau_h', 'saving', 'seed', 'file']
    directory = pd.DataFrame(index=list(range(len(file_list))), columns=cols)

    for i, file in enumerate(file_list):
        temp = file.split('_')
        directory.iloc[i, :] = [float(temp[1][1:]), float(temp[2][1:]),
                                float(temp[3][1:]), float(temp[5]),
                                float(temp[7]), float(temp[9]), float(temp[11]),
                                int(temp[12][2:]), int(temp[13][2:]),
                                int(temp[14][2:]), float(temp[15][3:]),
                                int(temp[16][4:6]), file]

    save_file = open(save_loc + 'directory.df', 'wb')
    pickle.dump(directory, save_file)
    save_file.close()


if __name__ == '__main__':
    params = {
        'tech0': np.exp(1), 'rho': 1 / 3, 'epsilon': 1e-5, 'tau_y': 1000,
        'dep': 0.0002,
        "tau_h": 25, "tau_s": 250, "c1": 1, "c2": 3e-4, "gamma": 2000,
        "beta1": 1.1, "beta2": 1.0, 'saving0': 0.15, "s0": 0, "h_h": 10
    }

    start = np.array([1, 10, 10, 0, 0, 1, params['saving0']])
    # Accurate production adjustment
    start[0] = params['epsilon'] + params['rho'] * min(start[1:3])

    params['c2'] = 2e-4
    sim_models(params, start, seeds=list(range(1, 20)),
               save_loc='long_run_approx/', verbose=True)

    params['c2'] = 4e-4
    sim_models(params, start, seeds=list(range(1, 20)),
               save_loc='long_run_approx/', verbose=True)

    params['c2'] = 3e-4
    params['gamma'] = 1000
    sim_models(params, start, seeds=list(range(1, 20)),
               save_loc='long_run_approx/', verbose=True)

    params['gamma'] = 3000
    sim_models(params, start, seeds=list(range(1, 20)),
               save_loc='long_run_approx/', verbose=True)
