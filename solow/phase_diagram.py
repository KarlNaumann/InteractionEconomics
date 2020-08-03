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
                              t_end: float = 2e5, args=None) -> list:
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
        args = args if args is not None else self.args + [self.ou]

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
                           plot: bool = True, sim_len: int = 2e5):

        prior_h_arg = self.h_args
        prior_ou = self.ou

        result = {
            'mean': pd.DataFrame(index=gammas, columns=sigmas),
            'std': pd.DataFrame(index=gammas, columns=sigmas),
            'down': pd.DataFrame(index=gammas, columns=sigmas),
            'mdd': pd.DataFrame(index=gammas, columns=sigmas)
        }

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
                info = self.point_classification(self.find_critical_points(),
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
                print(recessions)

                for id, val in analysis.items():
                    if id in result.keys():
                        result[id].iloc[g, sig] = val
        if plot:
            plt.tight_layout()
            plt.show()

        return result, recessions

    @staticmethod
    def recession_timing(gdp, timescale='Q-JAN'):
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
        growth = gdp.copy(deep=True)
        day_ix = pd.bdate_range('2000', freq='D', periods=gdp.shape[0])
        growth.index = day_ix
        growth = growth.resample(timescale, convention='end').agg(
                lambda x: (x[-1] - x[0]) / x[0])

        ix = growth < 0

        # expansion start => negative growth to two consecutive expansions
        expansions = np.flatnonzero(
                (ix.shift(-1) == False) & (ix == True) & (ix.shift(1) == True))

        # recession start => positive growth to two consecutive contractions
        recessions = np.flatnonzero(
                (ix == True) & (ix.shift(1) == False) & (ix.shift(2) == False))

        # Convert to indexes in original
        expansions = [day_ix.get_loc(growth.index[i]) for i in expansions]
        recessions = [day_ix.get_loc(growth.index[i]) for i in recessions]

        print('Expansion\t', expansions)
        print('Recession\t', recessions)

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

            """
            rec_ix = recessions.index(se[-1][1])
            while expansions[i] > recessions[rec_ix]:
                rec_ix += 1
            rec_ix = + 1
            if rec_ix >= len(recessions):
                break
            next_rec = recessions[rec_ix]
            if se[-1][1] < expansions[i] < next_rec:
                se.append((expansions[i], next_rec))
            i += 1
            """
        # remainders = [j for j in expansions if j>se[-1][1]]
        # if len(remainders)>0:
        #    se.append((remainders[0],))

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
