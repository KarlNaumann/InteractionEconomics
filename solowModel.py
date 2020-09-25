import pickle

import numpy as np
import pandas as pd
from cython_base.step_functions import long_general
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


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

    def visualise(self, asymp: bool = True, save: str = ''):

        # Checks
        assert self.path is not None, "Run simulation first"
        if self.asymp_rates is None:
            self.asymptotics()
        if self.sbars is None:
            self._s_sol()
        # Variables
        df = self.path.loc[:, ['y', 'ks', 'kd', 's', 'g']]
        t = df.index.values
        psi = self.asymp_rates[:3]
        # Figure setup
        fig, ax = plt.subplots(2, 2)
        fig.set_size_inches(8, 8)
        a_line = dict(linestyle='--', linewidth=1.25)
        n_line = dict(linestyle='-', linewidth=1.0)
        # 1 = Production
        if asymp:
            ax[0, 0].plot(t, t * psi[0] + df.y.iloc[0], label='Asymp. Growth',
                          color='orange', **a_line)
        ax[0, 0].plot(df.y, label='Production', color='royalblue', **n_line)
        ax[0, 0].set_title("Log Production (y)")
        ax[0, 0].set_xlabel("Time")
        ax[0, 0].set_ylabel("Log Production")
        ax[0, 0].set_xlim(0, t[-1])
        ax[0, 0].set_ylim(df.y.min(), df.y.max())
        ax[0, 0].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax[0, 0].legend(loc=2)
        # 2 = Capital Markets
        if asymp:
            ax[0, 1].plot(t, t * psi[1] + df.ks.iloc[0], label='Asymp. Supply',
                          color='orange', **a_line)
            ax[0, 1].plot(t, t * psi[2] + df.kd.iloc[0], label='Asymp. Demand',
                          color='gold', **a_line)
        ax[0, 1].plot(df.ks, label='Supply', color='royalblue', **n_line)
        ax[0, 1].plot(df.kd, label='Demand', color='skyblue', **n_line)
        ax[0, 1].set_title("Capital Markets (ks, kd)")
        ax[0, 1].set_xlabel("Time")
        ax[0, 1].set_ylabel("Log Capital")
        ax[0, 1].set_xlim(0, t[-1])
        ax[0, 1].set_ylim(min(df.ks.min(), df.kd.min()),
                          max(df.ks.max(), df.kd.max()))
        ax[0, 1].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax[0, 1].legend(loc=2)
        # 3 = Sentiment
        ax[1, 0].axhline(0, **a_line)
        if asymp:
            ax[1, 0].axhline(self.sbars[0], color='orange', **a_line)
            ax[1, 0].axhline(self.sbars[1], color='orange', **a_line)
        ax[1, 0].plot(df.s, label='Sentiment', color='royalblue', **n_line)
        ax[1, 0].set_title("Sentiment (s)")
        ax[1, 0].set_xlabel("Time")
        ax[1, 0].set_ylabel("Sentiment")
        ax[1, 0].set_xlim(0, t[-1])
        ax[1, 0].set_ylim(-1, 1)
        ax[1, 0].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax[1, 0].legend(loc=3)
        # 4 = Gamma Multiplier
        ax[1, 1].plot(df.g, label='Multiplier', color='royalblue', **n_line)
        ax[1, 1].set_title("Feedback Activation")
        ax[1, 1].set_xlabel("Time")
        ax[1, 1].set_ylabel("Multiplier")
        ax[1, 1].set_xlim(0, t[-1])
        ax[1, 1].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax[1, 1].legend(loc=3)
        # Final layout
        fig.tight_layout()
        plt.show()

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

    def _s_sol(self):
        """ Add the solution to the sentiment equilibria on the basis of the
        asymptotic growth rates

        Returns
        -------
        sbars = []
        """
        # Check
        if self.asymp_rates is None:
            self.asymptotics()
        # Solve
        m = self.params['beta2'] * np.tanh(
                self.params['gamma'] * self.asymp_rates[0])
        y = lambda s: np.abs(np.arctanh(s) - self.params['beta1'] * s - m)
        intersects = []
        # Search through four quadrants
        for bnds in [(-0.99, -0.5), (-0.5, 0), (0, 0.5), (0.5, 0.99)]:
            temp = minimize(y, 0.5 * sum(bnds), bounds=(bnds,), tol=1e-15)
            intersects.append(temp.x[0])
        self.sbars = [max(intersects), min(intersects)]
        return [max(intersects), min(intersects)]
