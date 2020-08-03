import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from numdifftools import Jacobian
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 10, 8
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['image.cmap'] = 'RdYlGn_r'

from ornstein_uhlenbeck import OrnsteinUhlenbeck


class PhaseDiagram(object):
    def __init__(self, tau_h: float = 25, tau_s: float = 250,
                 tau_y: float = 2000, epsilon: float = 5e-5, c1: float = 1,
                 c2: float = 15e-5, s0: float = 0, tech0: float = 1,
                 beta1: float = 1.1, beta2: float = 1.0, gamma: float = 2000,
                 theta: float = 0.2, sigma: float = 2.5):
        """ Class instance to generate various phase diagram representations of
        the dynamic capital demand system

        Parameters
        ----------
        tau_h, tau_s, tau_y   :   float (defaults: 25, 250, 2000)
            characteristic timescales of information, sentiment, and production
        epsilon :   float (default: 5e-5)
            technology growth rate
        c1, c2, s0  :   float (defaults: 1, 15e-5)
            coefficients of capital demand (change in sentiment, sentiment,
            long-term sentiment)
        tech0   :   float
            Initial level of technology
        beta1, beta2    :   float (defaults: 1.1, 1.0)
            coefficients of sentiment (interactions and information)
        gamma   :   float (default: 2000)
            strength of production feedback
        phi :   float (default: 2000)
            strength of the excess capital feedback
        theta, sigma    :   float (defaults: 0.2, 2.5)
            parametrisation of the Ornstein-Uhlenbeck white noise process
        """

        # Save the arguments to be used
        self.times = {"h": tau_h, "s": tau_s, "y": tau_y, "tech": 1 / epsilon, }
        self.epsilon = epsilon
        self.demand = {"c1": c1, "c2": c2, "s0": s0, 'tech0': tech0}
        self.s_args = {"beta1": beta1, "beta2": beta2, }
        self.h_args = {'gamma': gamma}
        self.theta = theta
        self.ou = OrnsteinUhlenbeck(
                **{'decay': theta, "drift": 0, "diffusion": sigma, "t0": 1})
        self.args = [self.times, self.demand, self.s_args, self.h_args]

        # Arguments for the critical value function
        self.crit_args = (beta1, beta2, gamma, c2, s0, epsilon)

    @staticmethod
    def velocity(t, x, times, demand, s_arg, h_arg, use_k=True,
                 ou=None) -> np.array:
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

        s, h, z = x[0], x[1], x[2]

        v_s = -s + np.tanh(s_arg['beta1'] * s + s_arg['beta2'] * h)
        v_s = v_s / times['s']

        delta_prod = (demand['tech0'] * np.exp(z) - 1) / times['y']
        force = h_arg['gamma'] * delta_prod
        if ou is not None: force += ou.euler_maruyama(t)

        v_h = (-h + np.tanh(force)) / times['h']

        v_z = demand['c1'] * v_s + demand['c2'] * s - delta_prod \
              + (1 / times['tech'])

        if use_k:
            v_k = demand['c1'] * v_s + demand['c2'] * s
            return np.array([v_s, v_h, v_z, v_k])
        else:
            return np.array([v_s, v_h, v_z])

    def find_critical_points(self) -> list:
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

        # Use minimiser to determine where the function crosses 0
        for x in np.linspace(-1, 1, 11):
            candidate = minimize(f, x0=x, bounds=[(-1, 1)], args=self.crit_args,
                                 method='L-BFGS-B', options=min_options)
            if candidate.success:
                sols.append(candidate.x[0])

        # Eliminate duplicated solutions (artefact of multiple x0 in minimise)
        filtered = []
        for i, val in enumerate(sols):
            found = False
            for j in range(i + 1, len(sols)):
                d = np.sum(np.abs(sols[j] - val))
                if d < 1e-7:
                    found = True
                    break
            if not found:
                filtered.append(val)

        # Determine h and z for each point
        coordinates = []
        for i, s in enumerate(filtered):
            inner = (self.demand['c2'] * (s - self.demand['s0']) + self.epsilon)
            h = np.tanh(self.h_args['gamma'] * inner)
            z = np.log((self.times['y'] * inner + 1) / self.demand['tech0'])
            coordinates.append((s, h, z))

        return coordinates

    def point_classification(self, crit_points: list,
                             plot: bool = True,
                             vector_field: bool = True) -> dict:

        result = {}

        # Lambda function to pass the arguments and t=0
        f = lambda x: self.velocity(0, x, self.times, self.demand, self.s_args,
                                    self.h_args, use_k=False, ou=None)

        # Iterate through points and categorize
        for point in crit_points:
            jacobian = Jacobian(f)(point)
            eig_val, eig_vec = np.linalg.eig(jacobian)
            result[point] = {
                'evec': eig_vec,
                'eval': eig_val
            }

            # Categorise Points
            if all(np.real(eig_val) < 0):
                result[point]['type'] = 'stable'
            else:
                result[point]['type'] = 'unstable'

        if plot:
            fig = plt.figure()
            fig.set_size_inches(15, 10)
            ax = fig.add_subplot(111)

            # Arrow arguments
            a_arg = {'headwidth': 3, 'width': 0.003}

            if vector_field:
                points = np.linspace(-1, 1, 11)
                s, h = np.meshgrid(points, points)
                z = np.zeros(s.shape)
                v_s, v_h, _ = self.velocity(0, [s, h, z], self.times,
                                            self.demand, self.s_args,
                                            self.h_args, use_k=False, ou=None)
                ax.quiver(s, h, v_s, v_h, width=0.001, headwidth=8,
                          color='gray')

            for x, info in result.items():
                # Plot arrows first, then overlay the solution points
                for i in range(info['evec'].shape[1]):
                    v = info['evec'][:, i] / np.linalg.norm(info['evec'][:, i])
                    v = v / 3
                    # Criteria for differentiating
                    eig_real = np.isreal(info['eval'][i])
                    eig_pos = np.real(info['eval'][i]) > 0

                    if eig_real and eig_pos:
                        ax.quiver(x[0], x[1], -v[0], -v[1], pivot='tail',
                                  color='blue', **a_arg)
                        ax.quiver(x[0], x[1], v[0], v[1], pivot='tail',
                                  color='blue', **a_arg)
                    elif eig_real and not eig_pos:
                        ax.quiver(x[0], x[1], -v[0], -v[1], pivot='tip',
                                  color='blue', **a_arg)
                        ax.quiver(x[0], x[1], v[0], v[1], pivot='tip',
                                  color='blue', **a_arg)
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
                if info['type'] == 'stable':
                    ax.scatter(x[0], x[1], c='green', label='stable')
                else:
                    ax.scatter(x[0], x[1], c='red', label='unstable')

            ax.legend()
            ax.set_xlabel("s")
            ax.set_ylabel("h")
            ax.set_xticks(np.linspace(-1, 1, 11))
            ax.set_yticks(np.linspace(-1, 1, 11))
            plt.show()

        return result

    def stochastic_simulation(self, init_val_list: list, t0: float = 1,
                              t_end: float = 2e5, args=None):
        """ Simulate the stochastically forced dynamical system in the
        (s,h,z)-space, i.e. under the condition that Kd<Ks for all periods

        Parameters
        ----------
        init_val_list   :   list
            Initial coordinates
        t0  :   float
            Starting point
        t_end   :   float
            How long to integrate for (in business days)
        args    :   tuple
            Optionally define own args tuple of dict (see init docstring)

        Returns
        -------
        path    :   list of df
            DataFrame of the path for each of the variables across starting
            points

        """
        paths = []
        args = self.args + [self.ou]

        # Gather solution candidates from different starting values
        for start in init_val_list:
            # Solve the initial value problem
            path = solve_ivp(self.velocity, t_span=(t0, t_end),
                             y0=np.array(start), vectorized=True,
                             t_eval=np.arange(int(t0), int(t_end) - 1),
                             max_step=1.0, method='RK45', args=args)
            # Grab output
            df = pd.DataFrame(path.y.T, columns=['s', 'h', 'z', 'k'])
            y = df.k - df.z + self.epsilon * np.arange(int(t0),
                                                       df.shape[0] + int(t0))
            df.loc[:, 'y'] = y
            paths.append(df)

        return paths

    def gamma_theta_phases(self, start: list, gammas: list, sigmas: list,
                           plot: bool = True):

        result = pd.DataFrame(index=gammas, columns=sigmas)

        for gamma in gammas:
            for sigma in sigmas:
                # Set up the params
                h_args = {'gamma': gamma}
                ou = OrnsteinUhlenbeck(
                        **{
                            'decay': self.theta, "drift": 0, "diffusion": sigma,
                            "t0": 1
                        })
                args = [self.times, self.demand, self.s_args, h_args, ou]

                # Simulate
                path = self.stochastic_simulation([start], t0=1, t_end=2e5,
                                                  args=args)
                recessions = self.recessionSE(np.exp(path[0].y), 63)

                result.loc[gamma, sigma] = np.mean(recessions.start.diff(1))

        if plot:
            f = {'fontsize': 12, 'fontweight': 'medium'}
            cmap = plt.get_cmap('OrRd')

            fig = plt.figure(constrained_layout=True)
            fig.set_size_inches((4, 10))

            ax = fig.add_subplot()
            sns.heatmap(result, annot=True, ax=ax)

            plt.show()

        return result

    @staticmethod
    def recessionSE(gdp, t):
        """ Calculate the start and end of recessions on the basis of gdp growth.
        Two consecutive periods of length t that have negative growth start a
        recession. Two with positive growth end it.

        Parameters
        ----------
        gdp
        t

        Returns
        -------

        """
        growth = gdp.pct_change(t)
        ix = growth < 0

        ix_next = ix.shift(1)
        ix_next2 = ix.shift(2)
        ix_prev = ix.shift(-1)

        df = pd.DataFrame(dict(
                # Start => go from negative growth to two consecutive expansions
                start=np.flatnonzero((ix.shift(-1) == False) & (ix == True) & (
                        ix.shift(1) == True)),
                # End => go from positive growth to two consecutive contractions
                end=np.flatnonzero((ix == True) & (ix.shift(1) == False) & (
                        ix.shift(2) == False))))

        return df
