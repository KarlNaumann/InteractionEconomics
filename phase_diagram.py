import numpy as np
import pandas as pd
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
                 phi: float = 1.0, theta: float = 0.2, sigma: float = 2.5):
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
        # Arguments on a per function basis
        self.z_args = {
            "tau": tau_y, "eps": epsilon, "c1": c1, "c2": c2, "s0": s0,
            "tech0": tech0
        }
        self.s_args = {"tau": tau_s, "b1": beta1, "b2": beta2}
        self.h_args = {"tau": tau_h, "gamma": gamma, "phi": phi}
        self.ou_args = {"decay": theta, "diffusion": sigma, "drift": 0, "t0": 1}

        # Initialise some additional useful params
        self.epsilon = epsilon
        self.ou = OrnsteinUhlenbeck(**self.ou_args)

        self._crit_point_info = None

    @staticmethod
    def velocity(t, x, z_args, s_args, h_args, use_k=True, ou=None) -> np.array:
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

        v_s = -s + np.tanh(s_args['b1'] * s + s_args['b2'] * h)
        v_s = v_s / s_args['tau']

        delta_prod = (z_args['tech0'] * np.exp(z) - 1) / z_args['tau']
        if ou is not None:
            force = h_args['gamma'] * delta_prod + ou.euler_maruyama(t)
        else:
            force = h_args['gamma'] * delta_prod

        v_h = (-h + np.tanh(force)) / h_args['tau']
        v_z = (z_args['c1'] * v_s) + (
                    z_args['c2'] * (s - z_args['s0'])) - delta_prod + z_args[
                  "eps"]

        if use_k:
            v_k = z_args['c1'] * v_s + z_args['c2'] * s
            return np.array([v_s, v_h, v_z, v_k])
        else:
            return np.array([v_s, v_h, v_z])

    def overview(self, start: np.ndarray, plot: bool = True,
                 t_end: float = 1e5, save: str = '') -> pd.DataFrame:
        """ Generate an overview of the system under the given parameters by
        (1) phase diagram, (2) Sentiment plot and analysis, (3) 3D plot

        Parameters
        ----------
        start   :   np.ndarray
            Starting point in the (s,h,z) space
        plot    :   boolean

        Returns
        -------
        path    :   pd.DataFrame
            DataFrame containing the
        """

        # First determine critical points and prep to plot phase Diagram
        critical_points = self._point_classification(self._critical_points())
        print("Critical points at:")
        for point, v in critical_points.items():
            print("({:.2f},{:.2f},{:.2f})\t-\t{}".format(*point, v['kind']))

        # Generate a simulation for the system
        path = self.simulate([start], t_end=t_end)

        # Generate the plots
        fig = plt.figure()
        fig.set_size_inches(8, 15)
        ax_lst = []

        # Phase Diagram
        ax_lst.append(fig.add_subplot(311))
        self.phase_diagram(show=False, ax=ax_lst[0], start=start)
        ax_lst[0].set_title("Phase Diagram in the (s,h) space")

        # Sentiment over time
        ax_lst.append(fig.add_subplot(312))
        self._sentiment_plot(path, ax=ax_lst[1])

        # Add production to the plot
        ax2 = ax_lst[1].twinx()
        ax2.plot(path.y, color='green')
        ax2.set_ylabel("Log Production (y)")

        ax_lst.append(fig.add_subplot(313))
        ax_lst[2].plot(path.s, path.z, color='blue', linewidth=0.5)
        ax_lst[2].set_xlabel("Sentiment (s)")
        ax_lst[2].set_ylabel("Z")
        ax_lst[2].set_xlim(-1, 1)
        for point, v in self._crit_point_info.items():
            if 'unstable' not in v['kind']:
                ax_lst[2].scatter(point[0], point[2], linewidths=1, color='red')
        # 3D plot of (s,h,z) space
        #ax_lst.append(fig.add_subplot(313, projection='3d'))
        #self._3d_trajectory(path, ax=ax_lst[2], rotate=False)

        plt.tight_layout()
        if save is not '':
            plt.savefig(save, bbox_inches='tight')
        plt.show(block=False)

    def _3d_trajectory(self, path: pd.DataFrame, ax=None, rotate: bool = False):
        """ 3D plot in the s,h,z space

        Parameters
        ----------
        ax  :   matplotlib.axes._subplots.Axes3DSubplot
        rotate  :   whether to animate a rotation

        Returns
        -------
        ax  :   matplotlib.axes._subplots.Axes3DSubplot
        """

        if ax is None:
            fig = plt.figure()
            fig.set_size_inches(15, 10)
            ax = fig.add_subplot(111, projection='3d')

        # Find and include the first two  business cycles
        crossings = (path.s > 0).astype(int).diff()
        cross_ix = [path.index[i] for i in range(crossings.shape[0]) if
                    crossings.iloc[i] != 0]

        # Detrend the series
        path = path.sub(path.mean())

        # Smooth the series

        ax.plot(path.h.rolling(int(1e2)).mean(),
                path.s.rolling(int(1e2)).mean(),
                path.z.rolling(int(1e2)).mean())
        for point, v in self._crit_point_info.items():
            if 'unstable' not in v['kind']:
                ax.scatter(point[1], point[0], point[2], linewidths=1,
                           color='red')
        ax.set_ylabel('Sentiment (s)')
        ax.set_xlabel('Information (h)')
        ax.set_zlabel('Z')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

        if rotate:
            # rotate the axes and update
            for angle in range(0, 360):
                ax.view_init(30, angle)
                plt.draw()
                plt.pause(.001)

        return ax

    def _sentiment_plot(self, path: pd.DataFrame, ax=None):
        """

        Parameters
        ----------
        path
        ax

        Returns
        -------

        """
        if ax is None:
            fig = plt.figure()
            fig.set_size_inches(15, 10)
            ax = fig.add_subplot(111)

        # Analysis of the sentiment
        crossings = (path.s > 0).astype(int).diff()
        cross_ix = [path.index[i] for i in range(crossings.shape[0]) if
                    crossings.iloc[i] != 0]
        avg_dur = np.mean(pd.Series(cross_ix[::2]).diff())
        std_dev = np.std(pd.Series(cross_ix[::2]).diff())

        ax.plot(path.s)
        ax.set_title("Sentiment and Production over time")
        ax.set_ylim(-1, 1)
        ax.set_ylabel("Sentiment (s)")

        info = "Mean {:.0f}\n Std. {:.0f}".format(avg_dur, std_dev)
        ax.text(0, 0.75, info, horizontalalignment='left',
                verticalalignment='bottom')

        # Equilibrium values
        for point, v in self._crit_point_info.items():
            if 'unstable' not in v['kind']:
                ax.axhline(point[0], color='red', linestyle='--')

        # Zero crossing
        ax.axhline(0, color='black', linestyle='--', linewidth=0.5)

        # Crossings of 0
        # for ix in cross_ix:
        # ax.axvline(ix, color='gray', linestyle='--', linewidth=0.5)

        return ax

    def _critical_points(self, args: tuple = None) -> list:
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
        standard_args = (
            self.s_args["b1"], self.s_args["b1"], self.h_args["gamma"],
            self.z_args["c2"], self.z_args["s0"], self.epsilon)
        args = args if args is not None else standard_args

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
            inner = (self.z_args["c2"] * (s - self.z_args['s0']) + self.epsilon)
            h = np.tanh(self.h_args['gamma'] * inner)
            z = np.log((self.z_args['tau'] * inner + 1) / self.z_args['tech0'])
            coordinates.append((s, h, z))

        return coordinates

    def _point_classification(self, crit_points: list) -> dict:

        result = {}

        # Lambda function to pass the arguments and t=0
        f = lambda x: self.velocity(0, x, self.z_args, self.s_args, self.h_args,
                                    use_k=False, ou=None)

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

    def phase_diagram(self, show: bool = True, ax=None, start=None):
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
        if ax is None:
            fig = plt.figure()
            fig.set_size_inches(15, 10)
            ax = fig.add_subplot(111)

        # Arrow arguments
        a_arg = {'headwidth': 3, 'width': 0.003}

        if start is not None:
            # Sample trajectories
            trajectories = {}
            z0 = start[-2]
            k0 = start[-1]
            for i in np.linspace(-0.8, 0.8, 9):
                x0 = [-1, i, z0, k0]
                path = self.simulate([x0], t_end=1e4, stochastic=False)
                ax.plot(path.s, path.h, color='blue', linewidth=0.5)
                x0 = [i, -1, z0, k0]
                path = self.simulate([x0], t_end=1e4, stochastic=False)
                ax.plot(path.s, path.h, color='blue', linewidth=0.5)
                x0 = [1, i, z0, k0]
                path = self.simulate([x0], t_end=1e4, stochastic=False)
                ax.plot(path.s, path.h, color='blue', linewidth=0.5)
                x0 = [i, 1, z0, k0]
                path = self.simulate([x0], t_end=1e4, stochastic=False)
                ax.plot(path.s, path.h, color='blue', linewidth=0.5)

        # Vector field
        points = np.linspace(-1, 1, 11)
        s, h = np.meshgrid(points, points)
        z = np.zeros(s.shape)
        v_s, v_h, _ = self.velocity(0, [s, h, z], self.z_args,
                                    self.s_args, self.h_args,
                                    use_k=False, ou=None)
        ax.quiver(s, h, v_s, v_h, width=0.001, headwidth=8,
                  color='gray')

        for x, info in self._crit_point_info.items():
            # Plot arrows first, then overlay the solution points
            for i in range(info['evec'].shape[1]):
                v = info['evec'][:, i] / np.linalg.norm(info['evec'][:, i]) / 3
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
            if "unstable" in info['kind']:
                ax.scatter(x[0], x[1], c='red', label=info['kind'])
            else:
                ax.scatter(x[0], x[1], c='green', label=info['kind'])

        ax.legend()
        ax.set_xlabel("s")
        ax.set_ylabel("h")
        ax.set_xticks(np.linspace(-1, 1, 11))
        ax.set_yticks(np.linspace(-1, 1, 11))
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        if show and ax is None:
            plt.show()

        return ax

    def stability_plot(self, gamma: np.ndarray,
                       epsilon: np.ndarray, plot: bool = True,
                       save: str = '') -> pd.DataFrame:

        """ Function to determine at which points either of the two equilibria
        become unstable

        Parameters
        ----------
        gammas, thetas  :   np.ndarray
            parameter space to iterate across
        plot    :   bool
        Returns
        -------

        """
        args = list(self.crit_args)

        df = pd.DataFrame(index=np.sort(gamma), columns=np.sort(epsilon))

        for i, g in enumerate(df.index):
            for j, e in enumerate(df.columns):
                args[2] = g
                args[-1] = e
                points = self._find_critical_points(args=tuple(args))
                info = self._point_classification(points, plot=False)
                count = [1 for i in info if info[i]['type'] == 'stable']
                df.iloc[i, j] = sum(count) == 2
        if plot:
            fig, ax = plt.subplots()
            fig.set_size_inches(10, 12)
            plt.imshow(df.astype(float).values, cmap='RdYlGn')
            ax.set_xlabel('Gamma')
            ax.set_xticks(np.linspace(0, df.shape[1], len(df.index[::5])))
            ax.set_xticklabels(['{:.2e}'.format(i) for i in df.index[::5]],
                               rotation='vertical')
            ax.set_ylabel('Epsilon')
            ax.set_yticks(np.linspace(0, df.shape[0], len(df.columns[::5])))
            ax.set_yticklabels(['{:.2e}'.format(i) for i in df.columns[::5]],
                               rotation='horizontal')
            if save != '':
                plt.savefig(save, bbox_inches='tight')
            plt.show()

        return df

    def simulate(self, init_val_list: list, t0: float = 1,
                 t_end: float = 2e5, args=None,
                 stochastic: bool = True) -> list:

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
        if stochastic:
            args = (self.z_args, self.s_args, self.h_args, True,
                    OrnsteinUhlenbeck(**self.ou_args))
        else:
            args = (self.z_args, self.s_args, self.h_args, True, None)

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

        if len(init_val_list) == 1:
            return paths[0]
        else:
            return paths

    def gamma_theta_phases(self, start: list, gammas: list, sigmas: list,
                           plot: bool = True, sim_len: int = 2e5):

        prior_h_arg = self.h_args
        prior_ou = self.ou

        result = {
            'mean': pd.DataFrame(index=gammas, columns=sigmas),
            'std': pd.DataFrame(index=gammas, columns=sigmas),
            'down': pd.DataFrame(index=gammas, columns=sigmas),
            'mdd': pd.DataFrame(index=gammas, columns=sigmas)
        }
        rec_dict = {}
        path_dict = {}

        if plot:
            fig, ax_lst = plt.subplots(len(gammas), len(sigmas))
            fig.set_size_inches(15, 10)

        for g, gamma in enumerate(gammas):
            for sig, sigma in enumerate(sigmas):
                # Set up the params
                self.h_args = {'gamma': gamma}
                self.ou = OrnsteinUhlenbeck(
                        **{
                            'decay': self.theta, "drift": 0, "diffusion": sigma,
                            "t0": 1
                        })
                args = [self.times, self.demand, self.s_args, self.h_args, True,
                        self.ou]

                # Simulate
                path = self.stochastic_simulation([start], t0=1, t_end=sim_len,
                                                  args=args)
                # Find recessions based on the quarterly growth
                gdp = np.exp(path[0].y)
                recessions = self.recession_timing(gdp)
                analysis = self.recession_analysis(recessions, gdp)
                info = self._point_classification(self._find_critical_points(),
                                                  plot=False)
                stable_points = [info[p]['type'] == 'stable' for p in
                                 info.keys()]
                stable = sum(stable_points) == 2

                if plot:
                    ax_lst[g, sig].plot(gdp)
                    for i in range(1, recessions.shape[0]):
                        ax_lst[g, sig].axvspan(recessions.iloc[i - 1, 1],
                                               recessions.iloc[i, 0],
                                               color='gray')
                    info = """Mean: {:.2f}\nStd: {:.2f}\nMDD: {:.2f}\n Down: {:.2f} \n Stable: {}""".format(
                            analysis['mean'], analysis['std'], analysis['mdd'],
                            analysis['down'], stable)
                    ax_lst[g, sig].text(0.15, 0.8, info, ha='center',
                                        va='center', fontsize=9,
                                        transform=ax_lst[g, sig].transAxes)
                    ax_lst[g, sig].set_title(
                            'Gamma: {:.2f}, Sigma: {:.2f}'.format(gamma, sigma))

                for id, val in analysis.items():
                    if id in result.keys():
                        result[id].iloc[g, sig] = val

                rec_dict[(gamma, sigma)] = recessions
                path_dict[(gamma, sigma)] = path

        if plot:
            plt.tight_layout()
            plt.show()

        # Return parameters to original level
        self.h_args = prior_h_arg
        self.ou = prior_ou

        return result, rec_dict, path_dict

    @staticmethod
    def recession_timing(gdp, timescale=63):
        """ Calculate the start and end of recessions on the basis of gdp growth.
        Two consecutive periods of length t that have negative growth start a
        recession. Two with positive growth end it.

        Parameters
        ----------
        gdp
        t

        Returns
        -------
        df  :   pd.DataFrame
            DataFrame containing full cycles in sample. I.e. start at first
            moment of expansion, end at last moment of expansions. Gives the
            expa

        """

        growth = gdp.copy(deep=True).iloc[::timescale].pct_change()
        ix = growth < 0

        # expansion start => negative growth to two consecutive expansions
        expansions = np.flatnonzero(
                (ix.shift(-1) == False) & (ix == True) & (ix.shift(1) == True))

        # recession start => positive growth to two consecutive contractions
        recessions = np.flatnonzero(
                (ix == True) & (ix.shift(1) == False) & (ix.shift(2) == False))

        # Convert to indexes in original
        expansions = [growth.index[i] for i in expansions]
        recessions = [growth.index[i] for i in recessions]

        # Generate start end tuples, first is given (assume we start in exp.)
        se = [(0, min(recessions))]

        i = 0
        while i < len(expansions):
            # Determine the index of the next recession
            rec_ix = min([ix for ix in recessions if ix > expansions[i]]
                         + [gdp.shape[0]])
            # Check if the recession is admissible
            if rec_ix > se[-1][1]:
                se.append((expansions[i], rec_ix))
            # Find expansion after this
            exp = min([i for i in expansions if i > rec_ix] + [gdp.shape[0]])
            if exp == gdp.shape[0]:
                break
            else:
                i = expansions.index(exp)

        return pd.DataFrame(se, columns=['expansion', 'recession'])

    @staticmethod
    def recession_analysis(rec, gdp) -> dict:
        """ Function to wrap some basic analysis of the recessions

        Parameters
        ----------
        rec  :   pd.DataFrame
            recessions with first col being expansion starts, and second
            recession starts
        gdp     :   pd.Series


        Returns
        -------
        analysis    :   dict
        """
        mdd, down, up = [], [], []

        # Find the max drawdown and the time to max drawdown
        for i in range(rec.shape[0] - 1):
            # expansion to expansion
            window = gdp.loc[rec.iloc[i, 0]:rec.iloc[i + 1, 0]]
            # recession to recession
            window2 = gdp.loc[rec.iloc[i, 1]:rec.iloc[i + 1, 1]]
            # Maximum drawdown during a cycle
            mdd.append(100 * (window.max() - window2.min()) / window.max())
            # Time from start of recession to trough
            down.append(window2.idxmin() - rec.iloc[i, 1])

        analysis = {
            'mean': np.mean(rec.expansion.diff(1)),
            'std': np.std(rec.expansion.diff(1)),
            'mdd': np.mean(mdd),
            'down': np.mean(down)
        }

        print(analysis)

        return analysis
