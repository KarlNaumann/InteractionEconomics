import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from numdifftools import Jacobian
from cython_base.step_functions import demand_case

import utilities as ut
ut.plot_settings()


class DemandSolow(object):
    def __init__(self, params:dict, xi_args:dict):
        """Implementation of the demand limiting case, where capital is set to
        capital demand across time.

        Attributes
        -----------
        params  :   dict
            dictionary of parameters, must contain [tech0, epsilon, rho, tau_y,
            tau_s, tau_h, c1, c2, b1, b2, gamma
        xi_args :   dict
            dictionary of parameters for the Ornstein-Uhlenbeck process
        path    :   pd.DataFrame
            simulated path for Y and K

        Methods
        -------
        simulate()      :   solve initial value problem
        visualise_y(i)  :   matplotlib plot of production

        """
        self.params = params
        if 's0' not in params.keys():
            self.params['s0'] = -0.1#-0.15
        self.xi_args = xi_args
        self.path = None

        self._crit_point_info = None


    def simulate(self, initial_values: np.ndarray, interval:float=0.1,
                 t_end: float=1e5, xi:bool=True, seed:int=42) -> pd.DataFrame:
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
        # Initialise and pre-compute random dW
        np.random.seed(seed)
        if xi:
            stoch = np.random.normal(0, 1, int(t_end / interval))
        else:
            stoch = np.zeros(int(t_end / interval))
        # Initialise array for the results
        values = np.zeros((int(t_end), 6), dtype=float)
        values[0, :] = initial_values
        # Simulation via Cython function
        path = demand_case(interval, int(1 / interval), stoch, values,
                           **self.params, **self.xi_args)
        # Output and save arguments
        self.seed = seed
        self.t_end = t_end
        self.path = pd.DataFrame(path,
                                columns=['y', 'z', 'kd', 's', 'h', 'xi'])
        return self.path


    def phase_diagram(self, save:str=''):
        assert self.path is not None, "Run simulation first"
        vals = self.path

        # Generate the plots
        fig, ax = plt.subplots(3, 1)
        fig.set_size_inches(8, 12)

        # Phase Diagram
        self._sh_phase(ax[0])
        ax[0].set_title("Phase Diagram in the (s,h) space")

        # Sentiment over time
        ax[1].plot(vals.s)
        ax[1].set_title("Sentiment and Production over time")
        ax[1].set_ylim(-1, 1)
        ax[1].set_ylabel("Sentiment (s)")
        ax[1].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        # Equilibrium values
        for point, v in self._crit_point_info.items():
            if 'unstable' not in v['kind']:
                ax[1].axhline(point[0], color='red', linestyle='--')
        ax[1].axhline(0, color='black', linestyle='--', linewidth=0.5)
        # Add production to the plot
        ax2 = ax[1].twinx()
        ax2.plot(vals.y, color='green')
        ax2.set_ylabel("Log Production (y)")
        # Sentiment versus Z
        ax[2].plot(vals.s, vals.z, color='blue', linewidth=0.5)
        ax[2].set_xlabel("Sentiment (s)")
        ax[2].set_ylabel("Z")
        ax[2].set_xlim(-1, 1)
        for point, v in self._crit_point_info.items():
            if 'unstable' not in v['kind']:
                ax[2].scatter(point[0], point[2], linewidths=1, color='red')

        plt.tight_layout()
        if save != '':
            if '.png' not in save:
                save += '.png'
            plt.savefig(save, bbox_inches='tight')
        else:
            plt.show()
        plt.close()


    def get_critical_points(self):
        x = self._critical_points()
        self._crit_point_info = self._point_classification(x)
        return self._crit_point_info


    def sh_phase(self, save:str=''):
        plt.rcParams.update({'text.usetex':True})
        # Generate the plots
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(6, 6)
        # Phase Diagram
        self._sh_phase(ax)
        info = (self.params['gamma'], self.params['c2'])
        ax.set_title(r'$\gamma={:.0f},~c_2={:.0e}$'.format(*info))
        plt.tight_layout()
        if save != '':
            if '.png' not in save:
                save += '.png'
            plt.savefig(save, bbox_inches='tight')
        else:
            plt.show()
        plt.rcParams.update({'text.usetex':False})


    def _sh_phase(self, ax):
        """ Plot the phase diagram for the system under consideration including
        vector arrows

        Parameters
        ----------
        show    :   bool

        Returns
        -------
        ax  :   plt.axes
            can be used for further graphing purposes
        """

        if self._crit_point_info is None:
            self._point_classification(self._critical_points())

        # Arrow arguments
        a_arg = {'headwidth': 3, 'width': 0.003}

        # Sample trajectories
        trajectories = {}
        if self.path is not None:
            start = self.path.iloc[0, :]
            z0, k0 = start['z'], start['kd']
        else:
            z0, k0 = 0, 1

        for i in np.linspace(-0.8, 0.8, 9):
            x0 = np.array([1, z0, k0, -1, i, 0])
            path = self.simulate(x0, t_end=1e4, xi=False)
            ax.plot(path.s, path.h, color='blue', linewidth=0.5)
            x0 = np.array([1, z0, k0, i, -1, 0])
            path = self.simulate(x0, t_end=1e4, xi=False)
            ax.plot(path.s, path.h, color='blue', linewidth=0.5)
            x0 = np.array([1, z0, k0, 1, i, 0])
            path = self.simulate(x0, t_end=1e4, xi=False)
            ax.plot(path.s, path.h, color='blue', linewidth=0.5)
            x0 = np.array([1, z0, k0, i, 1, 0])
            path = self.simulate(x0, t_end=1e4, xi=False)
            ax.plot(path.s, path.h, color='blue', linewidth=0.5)

        for x, info in self._crit_point_info.items():
            # Plot arrows first, then overlay the solution points
            for i in range(info['evec'].shape[1]):
                v = info['evec'][:, i] / np.linalg.norm(info['evec'][:, i]) / 3
                # Criteria for differentiating
                eig_real = np.isreal(info['eval'][i])
                eig_pos = np.real(info['eval'][i]) > 0

                if eig_real and eig_pos:
                    ax.quiver(x[0], x[1], -v[0], -v[1], pivot='tail',
                              color='black', **a_arg)
                    ax.quiver(x[0], x[1], v[0], v[1], pivot='tail',
                              color='black', **a_arg)
                elif eig_real and not eig_pos:
                    ax.quiver(x[0], x[1], -v[0], -v[1], pivot='tip',
                              color='black', **a_arg)
                    ax.quiver(x[0], x[1], v[0], v[1], pivot='tip',
                              color='black', **a_arg)
                elif not eig_real and eig_pos:
                    ax.quiver(x[0], x[1], np.real(v[0]) / 1.5,
                              np.real(v[1]) / 1.5, pivot='tail',
                              color='red')
                    ax.quiver(x[0], x[1], -np.real(v[0]) / 1.5,
                              -np.real(v[1]) / 1.5, pivot='tail',
                              color='red')
                elif not eig_real and not eig_pos:
                    ax.quiver(x[0], x[1], np.real(v[0]) / 1.5,
                              np.real(v[1]) / 1.5, pivot='tip',
                              color='green', **a_arg)
                    ax.quiver(x[0], x[1], -np.real(v[0]) / 1.5,
                              -np.real(v[1]) / 1.5, pivot='tip',
                              color='green', **a_arg)

            # Plot the solutions
            if "unstable" in info['kind']:
                ax.scatter(x[0], x[1], c='red', label=info['kind'])
            else:
                ax.scatter(x[0], x[1], c='green', label=info['kind'])

        ax.legend(ncol=3, loc=4)
        ax.set_xlabel("s")
        ax.set_ylabel("h")
        ax.set_xticks(np.linspace(-1, 1, 11))
        ax.set_yticks(np.linspace(-1, 1, 11))
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        return ax


    def _critical_points(self) -> list:
        """ Determine the critical points in the system, where v_s, v_h and v_z
        are equal to 0. Do this by substituting in for s, solving s, and then
        solving the remaining points.

        Returns
        -------
        coordinates :   list
            list of tuples for critical coordinates in (s,h,z)-space
        """

        # minimisation function to find where this holds i.e. = 0
        def f(s, b1, b2, gamma, c2, s0, epsilon):
            inner = gamma * (c2 * (s - s0) + epsilon)
            return np.sqrt((np.tanh(b1 * s + b2 * np.tanh(inner)) - s) ** 2)

        sols = []
        min_options = {'eps': 1e-10}
        p = self.params
        args = (p['beta1'], p['beta2'], p['gamma'], p['c2'],
                p['s0'], p['epsilon'])

        # Use minimiser to determine where the function crosses 0 (s_dot=0)
        for x in np.linspace(-1, 1, 11):
            candidate = minimize(f, x0=x, bounds=[(-1, 1)], args=args,
                                 method='L-BFGS-B', options=min_options)
            if candidate.success:
                # Check if this critical point is already in the solution list
                if all([np.abs(sol - candidate.x[0]) >= 1e-7 for sol in sols]):
                    sols.append(candidate.x[0])

        # Determine h and z for each critical point in s
        coordinates = []
        for i, s in enumerate(sols):
            inner = (p["c2"] * (s - p['s0']) + p['epsilon'])
            h = np.tanh(p['gamma'] * inner)
            z = np.log((p['tau_y'] * inner + 1) / p['tech0'])
            coordinates.append((s, h, z))

        return coordinates


    @staticmethod
    def _velocity(t:float, x:np.ndarray, p:dict, reduced:bool) -> np.ndarray:
        """ Calculate the velocity of the demand system in (s,h,z)-space.
        Function is static so that it can be used in the solve_ivp optimiser

        Parameters
        ----------
        x   :   list
            values of (s,h,z) in that order
        times, demand, s_arg, h_arg   :   dict
            dictionaries of the relevant parametrisation
        use_k   :   bool
            whether to calculate the derivative in k as well

        ou  :   OrnsteinUhlenbeck
            Ornstein-Uhlenbeck white noise process instance

        Returns
        -------
        v_x :   list
            velocities of s,h,z at that point in time
        """

        _, _, s, h, z = x
        v_y = (p['tech0'] * np.exp(z) - 1) / p['tau_y']
        v_h = (-h + np.tanh(p['gamma'] * v_y)) / p['tau_h']
        v_s = (-s + np.tanh(p['beta1'] * s + p['beta2'] * h))/ p['tau_s']
        v_k = p['c1'] * v_s + p['c2'] * s
        v_z = v_k - v_y + p['epsilon']
        if reduced:
            return np.array([v_s, v_h, v_z])
        else:
            return np.array([v_y, v_k, v_s, v_h, v_z])


    def _point_classification(self, crit_points: list) -> dict:

        result = {}
        # Lambda function to pass the arguments and t=0
        f = lambda x: self._velocity(0, [0,0,*x], self.params, reduced=True)
        # Iterate through points and categorize
        for point in crit_points:
            jacobian = Jacobian(f)(point)
            eig_val, eig_vec = np.linalg.eig(jacobian)
            result[point] = {'evec': eig_vec, 'eval': eig_val}
            # Nodes
            if all(np.isreal(eig_val)):
                if all(eig_val < 0):
                    result[point]['kind'] = 'stable node'
                elif all(eig_val > 0):
                    result[point]['kind'] = 'unstable node'
                else:
                    result[point]['kind'] = 'unstable saddle'
            elif np.sum(np.isreal(eig_val)) == 1:
                if all(np.real(eig_val) < 0):
                    result[point]['kind'] = 'stable focus'
                elif all(np.real(eig_val) > 0):
                    result[point]['kind'] = 'unstable focus'
                else:
                    result[point]['kind'] = 'unstable saddle-focus'

        self._crit_point_info = result

        return result
