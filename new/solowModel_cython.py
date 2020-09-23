import numpy as np
import pandas as pd

# Cython solver import
from cython_base.step_functions import full_general, long_general


class SolowModel(object):
    def __init__(self, params: dict, xi_args=None):
        """ Class for the dynamic Solow model"""

        # Parametrization
        if xi_args is None:
            xi_args = dict(decay=0.2, diffusion=1.0)
        self.params = params
        self.xi_args = xi_args
        # Arguments for later
        self.path = None
        self.sbars = None
        self.t_end = 0
        self.seed = 0
        self.asymptotic_rates = None

    def simulate(self, initial_values: np.ndarray, t_end: float,
                 interval: float = 0.1, seed:int=40) -> pd.DataFrame:

        self.seed = seed
        self.t_end = t_end

        # News process
        np.random.seed(seed)
        stoch = np.random.normal(0, 1, int(t_end / interval))

        values = np.zeros((int(t_end), 7), dtype=float)
        values[0, :] = initial_values

        path = long_general(interval, int(1/interval), stoch, values, **self.xi_args,
                            **self.params)

        cols = ['y', 'ks', 'kd', 's', 'h', 'g', 'news']
        # Sampling for every business day
        self.path = pd.DataFrame(path, columns=cols)

        return self.path




    def asymptotics(self):
        p = self.params
        df = self.path.loc[:, ['y', 'ks', 'kd', 's']]
        psi = ((df.iloc[-1, :] - df.iloc[0, :]) / df.shape[0]).values
        psi_y, psi_ks, psi_kd,_ = psi
        g = psi_kd / (p['c2'] * p['beta2'] * p['gamma'] * psi_y)
        sbar_hat = df.s.mean()
        sbar_theory = psi_kd / p['c2']
        sbar_crit = p['epsilon'] / (p['c2']*(1-p['rho']))

        return [psi_y, psi_ks, psi_kd, g, sbar_hat, sbar_theory, sbar_crit]

