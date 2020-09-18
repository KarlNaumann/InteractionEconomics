import numpy as np
import pandas as pd

# Cython solver import
from step_functions import full_general


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

    def simulate(self, initial_values: np.ndarray, t_end: float = 1e6,
                 interval: float = 0.1, seed: int = 42) -> pd.DataFrame:
        """ Use a Cython compiled step-iterator to solve along the path

        Parameters
        ----------
        initial_values  :   np.ndarray
            array of initial values, order: [y, ks, kd, s, h, g, news]
        t_end   :   float
            Duration of the simulation in business days
        interval    :   float
            Interval of simulation, fraction of a day
        seed    :   int
            random seed for numpy

        Returns
        -------
        path    :   pd.DataFrane
            path of SDE, same order as initial_values
        """

        self.seed = seed
        self.t_end = t_end

        # News process
        np.random.seed(seed)
        t_count = int(t_end / interval)
        stoch = np.random.normal(0, np.sqrt(interval), t_count)

        values = np.zeros((t_count, 7), dtype=float)
        values[0, :] = initial_values
        path = full_general(t_end, interval, stoch, values, **self.xi_args,
                            **self.params)

        cols = ['y', 'ks', 'kd', 's', 'h', 'g', 'news']
        # Sampling for every business day
        self.path = pd.DataFrame(path[::int(1 / interval), :], columns=cols)

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

