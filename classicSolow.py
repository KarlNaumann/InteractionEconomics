import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp


class ClassicSolow(object):
    def __init__(self, params):
        """Implementation of the classic Solow Model

        Attributes
        -----------
        params  :   dict
            dictionary of parameters, must contain [tech0, epsilon, rho, tau_y,
            saving0, dep
        path    :   pd.DataFrame
            simulated path for Y and K

        Methods
        -------
        simulate()      :   solve initial value problem
        visualise_y(i)  :   matplotlib plot of production

        """
        self.params = params
        self.path = None

    def simulate(self, initial_values: np.ndarray,
                 t_end: float) -> pd.DataFrame:
        """ Simulating the classic Solow model using Scipy with Runge-Kute
        4th order

        Parameters
        ----------
        initial_values  :   np.ndarray
            order is [Y, Ks]
        t_end   :   float
        interval    :   float
        seed    :   float

        Returns
        -------
        path    :   pd.DataFrame

        """
        p = self.params
        args = (
            p['tech0'], p['epsilon'], p['rho'], p['tau_y'],
            p['saving0'], p['dep']
        )
        t_eval = np.arange(1, int(t_end) - 1)
        path = solve_ivp(self._step, t_span=(1, t_end), y0=initial_values,
                         t_eval=t_eval, args=args)
        self.path = pd.DataFrame(path.y.T, columns=['Y', 'K'])
        return self.path

    @staticmethod
    def _step(t, x, tech0, e, rho, tau_y, saving0, dep):
        """ Single timestep in the classic solow model

        Parameters
        ----------
        x   :   list
            values at time t
        t   :   float
            time
        tech0, e, rho, tau_y, saving0, dep  :   float
            parametrisation of the model

        Returns
        -------
        v   :   list
            velocities at time t
        """
        v_y = (tech0 * np.exp(e * t) * (x[1] ** rho) - x[0]) / tau_y
        v_k = saving0 * x[0] - dep * x[1]
        return [v_y, v_k]

    def visualise(self, save: str = '', show: bool = True):
        """ Visual representation of production in the classic solow case

        Parameters
        ----------
        save    :   str
            filepath to save under (.png will be added if missing)
        show    :   bool
            whether or not to thow the figure
        Returns
        -------
        ax  :   matplotlib axis object
        """
        # Setup
        path = self.path
        # Determine inset dimensions
        mask = (path.Y - 2 * path.iloc[0, 0]).abs().argsort()[:2]
        ix = path.Y.iloc[mask].index[0]
        inset_xlim = [0, ]

        # Figure
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(8, 4)
        # Production
        ax[0].plot(path.Y)
        ax[0].set_xlabel('Time (t)')
        ax[0].set_ylabel('Production (Y)')
        ax[0].set_xlim(0, path.index[-1])
        # Inset
        axins_y = ax[0].inset_axes([0.1, 0.5, 0.4, 0.4])
        axins_y.plot(path.Y)
        axins_y.set_xlim(0, ix)
        y_min = path.Y.iloc[:ix].min() - 0.25*path.Y.iloc[0]
        axins_y.set_ylim(y_min, path.Y.iloc[ix])
        #axins_y.set_xticklabels('')
        #axins_y.set_yticklabels('')
        #a = axins_y.get_xticks().tolist()
        #a[1] = 'change'
        #axins_y.set_xticklabels(a)
        #axins_y.set_xticklabels(labels)
        #axins_y.ticklabel_format(style='sci', axis='x', scilimits=(0, 1))
        ax[0].indicate_inset_zoom(axins_y)
        #ax[0].ticklabel_format(style='sci', axis='x', scilimits=(0, 1))
        # Capital
        ax[1].plot(path.K)
        ax[1].set_xlabel('Time (t)')
        ax[1].set_ylabel('Capital (K)')
        ax[1].set_xlim(0, path.index[-1])
        # Inset
        axins_k = ax[1].inset_axes([0.1, 0.5, 0.4, 0.4])
        axins_k.plot(path.K)
        axins_k.set_xlim(0, ix)
        axins_k.set_ylim(path.K.iloc[:ix].min(), path.K.iloc[ix])
        #axins_k.set_xticklabels('')
        #axins_k.set_yticklabels('')
        axins_k.ticklabel_format(style='sci', axis='both', scilimits=(0, 1))
        ax[1].indicate_inset_zoom(axins_k)
        ax[1].ticklabel_format(style='sci', axis='both', scilimits=(0, 1))
        # Complete
        fig.tight_layout()
        if save != '':
            if '.png' not in save:
                save += '.png'
            plt.savefig(save, bbox_inches='tight')
        plt.show()
        return ax
