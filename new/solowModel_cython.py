import pickle

import numpy as np
import pandas as pd
from cython_base.step_functions import long_general


class SolowModel(object):
    def __init__(self, params: dict, xi_args=None):
        """ Class for the dynamic Solow model"""

        # Parametrization
        if xi_args is None:
            xi_args = dict(decay=0.2, diffusion=1.0)
        self.params = params
        self.xi_args = xi_args
        # Arguments for later
        self.path, self.sbars, self.t_end, self.seed = None, None, None, None
        self.asymp_rates = None

    def simulate(self, initial_values: np.ndarray, t_end: float,
                 interval: float = 0.1, seed: int = 40) -> pd.DataFrame:
        """ Simulation of the dynamic solow model, wrapping a Cython
        implementation of the main integration function. Maximal simulation size
        depends on the amount of RAM available.

        Parameters
        ----------
        initial_values  :   np.ndarray
            Order of variables: [y, ks, kd, s, h, g, xi]
        t_end           :   float
            Duration of the simulation, recommend >1e6, 1e7 takes ~11 sec
        interval        :   float
            Integration interval, 0.1 by default
        seed            :   int
            Numpy random seed, 40 by default

        Returns
        -------
        path    :   pd.DataFrame
            DataFrame of the integration path indexed by t

        """
        # Initialise and pre-compute random dW
        np.random.seed(seed)
        stoch = np.random.normal(0, 1, int(t_end / interval))
        # Initialise array for the results
        values = np.zeros((int(t_end), 7), dtype=float)
        values[0, :] = initial_values
        # Simulation via Cython function
        path = long_general(interval, int(1 / interval), stoch, values,
                            **self.xi_args, **self.params)
        # Output and save arguments
        self.seed = seed
        self.t_end = t_end
        self.path = pd.DataFrame(path,
                                 columns=['y', 'ks', 'kd', 's', 'h', 'g', 'xi'])
        return self.path

    def asymptotics(self):
        """ Empirically calculate the asymptotic growth rates, and average
        sentiment levels

        Returns
        -------
        asymptotics :   list
            order of params [psi_y, psi_ks, psi_kd, g, sbar_hat,
            sbar_theory, sbar_crit]
        """
        # Initialisation
        assert self.path is not None, "Simulation run required first"
        df = self.path.loc[:, ['y', 'ks', 'kd', 's']]
        p = self.params
        # Asymptotic growth rates
        psi = ((df.iloc[-1, :] - df.iloc[0, :]) / df.shape[0]).values
        psi_y, psi_ks, psi_kd, _ = psi
        # Constant G
        g = psi_kd / (p['c2'] * p['beta2'] * p['gamma'] * psi_y)
        # Average sentiment values
        sbar_hat = df.s.mean()
        sbar_t = psi_kd / p['c2']
        sbar_c = p['epsilon'] / (p['c2'] * (1 - p['rho']))
        # Result
        self.asymp_rates = [psi_y, psi_ks, psi_kd, g, sbar_hat, sbar_t, sbar_c]
        return self.asymp_rates

    def save(self, item: str = 'model', folder: str = 'computations'):
        """ Export the model, or an aspect of the model, to a folder as a
        pickle object

        Parameters
        ----------
        item    :   str
            currently supported "model" for object, and "path" for dataframe
        folder  :   str
        """
        item = dict(model=[self, '.obj'], path=[self.path, '.df'])[item]
        file = open(self._name(folder) + item[1], 'wb')
        pickle.dump(item[0], file)
        file.close()

    def _name(self, folder: str = 'computations/') -> str:
        """ Auto-generate a filename for the model based on its parameters

        Parameters
        ----------
        folder  :str

        Returns
        -------
        name    :str
        """
        p = self.params
        name = '_'.join([
            'general', 't{:05.0e}'.format(self.t_end),
            'g{:05.0f}'.format(p['gamma']),
            'e{:07.1e}'.format(p['epsilon']),
            'c1_{:03.1f}'.format(p['c1']), 'c2_{:07.1e}'.format(p['c2']),
            'b1_{:03.1f}'.format(p['beta1']), 'b2_{:03.1f}'.format(p['beta2']),
            'ty{:03.0f}'.format(p['tau_y']), 'ts{:03.0f}'.format(p['tau_s']),
            'th{:02.0f}'.format(p['tau_h']),
            'lam{:01.2f}'.format(p['saving0']), 'dep{:07.1e}'.format(p['dep']),
            'tech{:04.2f}'.format(p['tech0']), 'rho{:04.2f}'.format(p['rho']),
        ])
        return folder + name
