import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numdifftools import Jacobian
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from capital_market import CapitalMarket
from firm import Firm
from household import Household
from ornstein_uhlenbeck import OrnsteinUhlenbeck


class SolowModel(object):
    def __init__(self, hh_kwargs: dict, firm_kwargs: dict, capital_kwargs: dict,
                 epsilon: float, ou_kwargs: dict, clearing_form: str = 'min',
                 v_excess: bool = True):
        """ Class implementing the Solow model

        Parameters
        ----------
        hh_kwargs       :   dict
            Keyword dictionary for the households. Format (see household docs):
            {   savings_rate: float,
                static: bool = True,
                dynamic_kwargs: dict = {} }
        firm_kwargs     :   dict
            Keyword dictionary for the firm. Format (see firm docs):
            {   prod_func: str = 'cobb-douglas',
                parameters: dict = {} }
        capital_kwargs  :   dict
            Keyword dictionary for the firm. Format (see firm docs):
            {   static: bool = True,
                depreciation: float = 0.2,
                pop_growth: float = 0.005
                dynamic_kwargs: dict = {
                    tau_s:float, beta1: float, beta2: float, tau_h:float,
                    gamma:float, phi:float, c1: float, c2: float, c3: float,
                } }
        ou_kwargs   :   dict
            Keyword dictionary for the Ornstein Uhlenbeck process. Format:
            {   decay: float, drift: float, diffusion: float, t0: float }
        epsilon   :   float
            technology growth rate
        clearing_form   :   str
            type of market clearing
        v_excess    :   bool
            Whether to calculate the velocity of the excess


        Returns
        ----------
        SolowModel instance
        """
        # Entities and Variables
        self.epsilon = epsilon
        self.firm = Firm(**firm_kwargs)
        self.household = Household(**hh_kwargs)
        self.cm = CapitalMarket(v_excess=v_excess, **capital_kwargs)
        self.ou_process = OrnsteinUhlenbeck(**ou_kwargs)

        # Storage
        self.path = None

    def solve(self, initial_values: dict, t0: float = 0, t_end: float = 1e2,
              stoch: bool = True):
        """ Iterate through the Solow model to generate a path for output,
        capital (supply & demand), and sentiment

        Parameters
        ----------
        initial_values  :   dict
            initial value dictionary, must include 'y','ks','tech'. Can include
            'kd','s','h'
        t0
        t_end

        Returns
        -------

        """
        # Agents involved in the economy
        entities = {
            'household': self.household,
            'firm': self.firm,
            'cap_mkt': self.cm,
        }
        if stoch:
            args = (entities, self.ou_process, self.epsilon)
        else:
            args = (entities, None, self.epsilon)

        # Solve the initial value problem
        path = solve_ivp(self._step, t_span=(t0, t_end), y0=initial_values,
                         max_step=1.0, method='RK45',
                         t_eval=np.arange(int(t0), int(t_end) - 1), args=args)

        self.path = path
        self.vars = self._path_df(path.y, t0, t_end)

        # Errors
        print(path.message)

        return self._path_df(path.y, t0, t_end)

    def _step(self, t, values: list, entities: dict,
              ou_process: OrnsteinUhlenbeck,
              epsilon: float):

        # Unpack inputs
        y, ks, kd, s, h, tech, cons, excess, r = values

        v_tech = epsilon * tech

        if ou_process is not None:
            news = ou_process.euler_maruyama(t)
        else:
            news = 0

        # Determine consumption and investment of hh at time t
        consumption, investment = entities['household'].consumption(y)
        v_cons = consumption - cons

        # Capital level given the supply and demand
        k = min(ks, np.exp(kd))
        v_e = -excess + np.log(ks) - kd

        # Determine new production level and velocity of production
        factors = {'k': k, 'n': 1, 'tech': tech}
        y_new, k_ret = entities['firm'].production(factors)
        v_y = entities['firm'].production_velocity(curr=y, new=y_new)
        v_r = -r + k_ret  # entities['cap_mkt'].eff_earnings(k, ks, k_ret)

        # Capital supply ( - depreciation + new investment)
        v_ks = entities['cap_mkt'].v_supply(ks, investment)

        # Determine the velocity of capital demand
        v_ln_y = entities['firm'].ln_vel(tech, k, y)  # v_y / y  #
        if entities['cap_mkt'].dyn_demand:
            temp = (np.log(ks) - kd)  # /ks
            # temp = (ks - np.exp(kd))/min([ks,np.exp(kd)])
            temp = max([(ks - np.exp(kd)) / ks, 0])
            v_kd, v_s, v_h = entities['cap_mkt'].v_demand(s, h, news, v_ln_y,
                                                          temp)
        else:
            v_kd, v_s, v_h = [0, 0, 0]

        return [v_y, v_ks, v_kd, v_s, v_h, v_tech, v_cons, v_e, v_r]

    def _log_step(self, t, values: list, y_args: dict, cap_args: dict,
                  gamma: float, h_k: float = 10, h_h: float = 10,
                  ou_process=None):
        """

        Parameters
        ----------
        t           :   float
        values      :   list
            values at time t, order: y, ks, kd, s, h, r
        y_args      :   dict
            arguments for production, incl. tech0, rho, e, tau_y, dep
        cap_args    :   dict
            arguments for capital mkt, incl. saving, depreciation, tau_h, tau_s,
            c1, c2, beta1, beta2
        gamma       :   float
        heaviside   :   float
        ou_process  :   OrnsteinUhlenbeck

        Returns
        -------
        velocities  :   list
        """
        y, ks, kd, s, h, gamma_mult = values

        if ou_process is not None:
            news = ou_process.euler_maruyama(t)
        else:
            news = 0

        k = kd - (kd - ks) * 0.5 * (1 + np.tanh(h_k * (kd - ks)))

        # Production changes
        z = y_args['rho'] * k + y_args['e'] * t - y
        v_y = (np.log(y_args['tech0']) * np.exp(z) - 1) / y_args['tau']

        # Capital supply changes
        v_ks = (cap_args['saving'] * np.exp(y) - y_args['dep'] * np.exp(ks))
        v_ks = v_ks / np.exp(ks)

        # Capital demand changes
        gamma_mult_n = 0.5 * (1 + np.tanh(h_h * (ks - kd)))
        v_g_s = gamma_mult_n - gamma_mult
        v_h = (-h + np.tanh(gamma * gamma_mult_n * v_y + news)) / cap_args['tau_h']

        force_s = cap_args['beta1'] * s + cap_args['beta2'] * h
        v_s = (-s + np.tanh(force_s)) / cap_args['tau_s']

        v_kd = cap_args['c1'] * v_s + cap_args['c2'] * (s - cap_args['s0'])

        return [v_y, v_ks, v_kd, v_s, v_h, v_g_s]

    def overview(self, start: list, y_args: dict, cap_args: dict,
                 gamma: float, h_k: float = 1e1, h_h: float = 1e1,
                 theta: float = 0.2, sigma: float = 1, t_end: float = 1e5,
                 save: str = ''):

        """ Develop exploratory graphics for a set of parameters

        Parameters
        ----------
        start       :   list
            initial values, order = y, ks, kd, s, h, r
        y_args      :   dict
            arguments for production, incl. tech0, rho, e, tau_y, dep
        cap_args    :   dict
            arguments for capital mkt, incl. saving, depreciation, tau_h, tau_s,
            c1, c2, beta1, beta2
        gamma       :   float
            Feedback strength
        heaviside   :   float
            Extent to which tanh will approximate the min()
        theta,sigma :   float
            Decay and diffusion of the Ornstein Uhlenbeck process
        t_end       :   float
        save        :   str
        """

        # Arguments
        t_eval = np.arange(1, int(t_end) - 1)
        ou = OrnsteinUhlenbeck(theta, sigma, 0)
        args = (y_args, cap_args, gamma, h_k, h_h, ou)

        # Generate path of variables
        path = solve_ivp(self._log_step, t_span=(1, t_end), y0=start,
                         method='RK45', t_eval=t_eval, args=args)
        print(path.message)
        df = pd.DataFrame(path.y.T, columns=['y', 'ks', 'kd', 's', 'h', 'g'])

        k = df.kd - (df.kd - df.ks) * 0.5 * (1 + np.tanh(h_k * (df.kd - df.ks)))
        e = pd.Series(np.arange(df.index[0], df.index[-1])) * y_args['e']
        z = y_args['rho'] * k + e - df.y

        # Generate a Figure
        fig, ax_lst = plt.subplots(2,3)
        fig.set_size_inches(12, 8)
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)

        # Production
        ax_lst[0, 0].plot(df.y)
        ax_lst[0, 0].set_title("Log Production (y)")
        ax_lst[0, 0].set_xlabel("Time")
        ax_lst[0, 0].set_ylabel("Log Production")
        ax_lst[0, 0].set_xlim(0, df.index[-1])

        # Sentiment
        ax_lst[0, 1].plot(df.s)
        ax_lst[0, 1].set_title("Sentiment (s)")
        ax_lst[0, 1].set_xlabel("Time")
        ax_lst[0, 1].set_ylabel("Sentiment (s)")
        ax_lst[0, 1].set_xlim(0, df.index[-1])

        # Capital Markets
        ax_lst[1, 0].plot(df.ks, label='Ks', color='Blue')
        ax_lst[1, 0].plot(df.kd, label='Kd', color='Red')
        ax_lst[1, 0].set_title("Capital Markets (ks, kd)")
        ax_lst[1, 0].set_xlabel("Time")
        ax_lst[1, 0].set_ylabel("Log Capital")
        ax_lst[1, 0].set_xlim(0, df.index[-1])
        ax_lst[1, 0].legend()

        # Feedback strength
        ax_lst[1, 1].plot(df.g)
        ax_lst[1, 1].set_title("Feedback Strength Multiplier")
        ax_lst[1, 1].set_xlabel("Time")
        ax_lst[1, 1].set_ylabel("Multiplier")
        ax_lst[1, 1].set_xlim(0, df.index[-1])

        # Limit cycles in z
        ax_lst[0, 2].plot(df.s, z)
        ax_lst[0, 2].set_title("Z")
        ax_lst[0, 2].set_xlabel("Sentiment")
        ax_lst[0, 2].set_ylabel("Z")
        ax_lst[0, 2].set_xlim(-1, 1)

        # Starting point information
        texts = ['y0: {:.1f}, '.format(start[0]),
                 's0: {:.1f}, h0: {:.1f}'.format(start[3],start[4]),
                 'ks0: {:.1f}, kd0: {:.1f}'.format(start[1],start[2]),
                 'gamma:{:.1e}'.format(gamma),
                 'h_h: {:.1e}'.format(h_h)]
        ax_lst[1, 2].text(0.5, 0.5, '\n'.join(texts),
                          transform=ax_lst[1, 2].transAxes, fontsize=14,
                          verticalalignment='center',
                          horizontalalignment='center', bbox=props)
        ax_lst[1, 2].set_axis_off()

        plt.tight_layout()
        if save is not '':
            plt.savefig(save, bbox_inches='tight')
        plt.show()

        return path

    def _critical_points(self, start: list, y_args: dict, cap_args: dict,
                         gamma: float, phi: float):

        """ Determine the critical points in the system, where v_s, v_h and v_z
        are equal to 0. Do this by substituting in for s, solving s, and then
        solving the remaining points.

        Returns
        -------
        coordinates :   list
            list of tuples for critical coordinates in (s,h,z)-space
        """

        args = (y_args, cap_args, gamma, phi)
        min_options = {'eps': 1e-10}
        bnds = [(-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf),
                (-1, 1), (-1, 1), (-np.inf, np.inf)]

        def f(point):
            v = self._log_step(1, point, *args)
            return sum([np.sqrt(v[3] ** 2), np.sqrt(v[4] ** 2)])

        solutions = []

        # Find the critical points
        for s in np.linspace(-0.9, 0.9, 7):
            for h in np.linspace(-0.9, 0.9, 7):
                start[3], start[4] = s, h
                candidate = minimize(f, x0=np.array(start), bounds=bnds,
                                     method='L-BFGS-B', options=min_options)
                if candidate.success:
                    # Check if this critical point is already in the solution list
                    if all([np.sum(np.abs(sol - candidate.x)) >= 1e-5 for sol in
                            solutions]):
                        solutions.append(candidate.x)

        return solutions

    def _point_classification(self, crit_points: list) -> dict:

        result = {}

        # Lambda function to pass the arguments and t=0
        entities = {
            'household': self.household,
            'firm': self.firm,
            'cap_mkt': self.cm
        }
        f = lambda x: np.array(self._step(0, x, entities, None, self.epsilon))

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

    def _initial_values(self, given: dict):

        """ Function to manipulate an initial value dictionary for compatibility
        with the solver

        Parameters
        ----------
        given   :   dict
            dictionary of initial values. Must include at least: 'y','ks','tech'

        Returns
        -------
        initial_values  :   list
            list of initial values for the solver.
            Order: y, ks, kd, s, h, tech, cons, excess, r
        """

        initial_values = [given['y'], given['ks']]
        if self.cm.dyn_demand:
            try:
                initial_values.extend([given['kd'], given['s'], given['h']])
            except KeyError:
                print("For dynamical demand system, provide kd, s and h")
            excess = max([given['ks'] - np.exp(given['kd']), 0])
        else:
            initial_values.extend([np.log(given['ks']), 0, 0])
            excess = 0

        initial_values.append(given['tech'])
        initial_values.append(self.household.savings_rate * given['y'])
        initial_values.append(excess)
        # Effective interest rate
        k = min([given['ks'], np.exp(given['kd'])])
        _, k_ret = self.firm.production({'k': k, 'tech': given['tech']})
        r = (k / given['ks']) * (
                k_ret - self.cm.depreciation) - self.cm.pop_growth
        initial_values.append(k_ret)
        return initial_values

    @staticmethod
    def _path_df(array, t0, t_end):
        """Convert the numpy array outputted by the solve_ivp to a pandas
        dataframe

        Parameters
        ----------
        array   :   np.array
            rows are the variables and columns are the time

        Returns
        -------
        df  :   pd.DataFrame
        """

        cols = ['y', 'ks', 'kd', 's', 'h', 'tech', 'cons', 'excess', 'r']
        # index = np.arange(int(t0), int(t_end) - 1)
        return pd.DataFrame(array.T, columns=cols)  # , index=index)

    def visualise(self, save: str = '', case: str = 'general'):
        """ Generate a plot to visualise the outputs"""
        df = self.vars.copy(deep=True)
        df.loc[:, 'kd'] = np.exp(df.loc[:, 'kd'])
        g_ix = df.loc[:, 'y'].pct_change() < 0
        s, e = self._recessionSE(g_ix)

        if case == 'general':
            fig, ax_lst = plt.subplots(4, 2)
            fig.set_size_inches(15, 10)
            k = df.loc[:, ['kd', 'ks']].min(axis=1)

            # Production and growth
            ax_lst[0, 0].plot(df.loc[:, 'y'])
            ax_lst[0, 0].set_title('Production')
            for j in zip(s, e):
                ax_lst[0, 0].axvspan(xmin=j[0], xmax=j[1], color='gainsboro')
            ax_lst[0, 1].plot(df.loc[:, 'y'].pct_change())
            ax_lst[0, 1].set_title('Production Growth Rates')
            for j in zip(s, e):
                ax_lst[0, 1].axvspan(xmin=j[0], xmax=j[1], color='gainsboro')

            # Capital Markets
            ax_lst[1, 0].plot(df.kd, label='Demand')
            ax_lst[1, 0].plot(df.ks, label='Supply')
            ax_lst[1, 0].legend()
            ax_lst[1, 0].set_title('Capital Supply and Demand')
            for j in zip(s, e):
                ax_lst[1, 0].axvspan(xmin=j[0], xmax=j[1], color='gainsboro')

            ax_lst[1, 1].plot(k)
            ax_lst[1, 1].set_title('Capital Applied to Production')

            # Variables of interest
            ax_lst[2, 0].plot(df.loc[:, 'r'])
            ax_lst[2, 0].set_title('Capital Earnings Rate from Production')
            ax_lst[2, 0].axhline(self.cm.depreciation)
            for j in zip(s, e):
                ax_lst[2, 0].axvspan(xmin=j[0], xmax=j[1], color='gainsboro')

            ax_lst[2, 1].plot(self.eff_interest(df.ks, k, df.r))
            ax_lst[2, 1].set_title('Effective Earnings')
            ax_lst[2, 1].axhline(0)
            for j in zip(s, e):
                ax_lst[2, 1].axvspan(xmin=j[0], xmax=j[1], color='gainsboro')

            # Capital Demand Dynamics
            ax_lst[3, 0].plot(df.loc[:, 's'])
            ax_lst[3, 0].set_title('Sentiment')
            for j in zip(s, e):
                ax_lst[3, 0].axvspan(xmin=j[0], xmax=j[1], color='gainsboro')
            ax_lst[3, 1].plot(df.loc[:, 'h'])
            ax_lst[3, 1].set_title('Information')

        elif case == 'ks':
            fig, ax_lst = plt.subplots(3, 1)
            fig.set_size_inches(15, 10)

            # Production and growth
            ax_lst[0].plot(df.loc[:, 'y'])
            ax_lst[0].set_title('Production')
            ax_lst[1].plot(df.loc[:, 'y'].pct_change())
            ax_lst[1].set_title('Production Growth Rates')

            # Capital Markets
            ax_lst[2].plot(df.ks, label='Supply')
            ax_lst[2].set_title('Capital Supply')

        elif case == 'kd':
            fig, ax_lst = plt.subplots(5, 1)
            fig.set_size_inches(15, 10)

            # Production and growth
            ax_lst[0].plot(df.loc[:, 'y'])
            ax_lst[0].set_title('Production')
            for j in zip(s, e):
                ax_lst[0].axvspan(xmin=j[0], xmax=j[1], color='gainsboro')
            ax_lst[1].plot(df.loc[:, 'y'].pct_change())
            ax_lst[1].set_title('Production Growth Rates')
            for j in zip(s, e):
                ax_lst[1].axvspan(xmin=j[0], xmax=j[1], color='gainsboro')

            # Capital Markets
            ax_lst[2].plot(df.kd, label='Demand')
            ax_lst[2].set_title('Capital Demand')

            # Capital Demand Dynamics
            ax_lst[3].plot(df.loc[:, 's'])
            ax_lst[3].set_title('Sentiment')
            ax_lst[4].plot(df.loc[:, 'h'])
            ax_lst[4].set_title('Information')

        elif case == 'overview':
            fig, ax_lst = plt.subplots(4, 1)
            fig.set_size_inches(10, 10)

            ax_lst[0].plot(np.log(df.loc[:, 'y']))
            ax_lst[0].set_title('Log Production')
            for j in zip(s, e):
                ax_lst[0].axvspan(xmin=j[0], xmax=j[1], color='gainsboro')

            ax_lst[1].plot(df.kd, label='Demand')
            ax_lst[1].plot(df.ks, label='Supply')
            ax_lst[1].legend()
            ax_lst[1].set_title('Capital Supply and Demand')
            for j in zip(s, e):
                ax_lst[1].axvspan(xmin=j[0], xmax=j[1], color='gainsboro')

            ax_lst[2].plot(df.loc[:, 's'])
            ax_lst[2].set_title('Sentiment')
            for j in zip(s, e):
                ax_lst[2].axvspan(xmin=j[0], xmax=j[1], color='gainsboro')

            ax_lst[3].plot(df.loc[:, 'h'])
            ax_lst[3].set_title('Information')

            for ax in ax_lst:
                ax.set_xlim(df.index[0], df.index[-1])

        plt.tight_layout()

        if save is not '':
            plt.savefig(save, bbox_inches='tight')
        plt.show()

    def eff_interest(self, ks, k, r):
        return (k.divide(ks) * r) - self.cm.depreciation

    def _recessionSE(self, ds):
        """returns list of (startdate,enddate) tuples for recessions"""
        start, end = [], []
        # Check to see if we start in recession
        if ds.iloc[0] == 1: start.extend([ds.index[0]])
        # add recession start and end dates
        for i in range(1, ds.shape[0]):
            a = ds.iloc[i - 1]
            b = ds.iloc[i]
            if a == 0 and b == 1:
                start.extend([ds.index[i]])
            elif a == 1 and b == 0:
                end.extend([ds.index[i - 1]])
        # if there is a recession at the end, add the last date
        if len(start) > len(end): end.extend([ds.index[-1]])
        return start, end

    def recession_timing(self, gdp: pd.Series, timescale=63):
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

        growth = gdp.iloc[::timescale].pct_change()
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
            'min': np.min(rec.expansion.diff(1)),
            'mean': np.mean(rec.expansion.diff(1)),
            'max': np.max(rec.expansion.diff(1)),
            'std': np.std(rec.expansion.diff(1)),
            'mdd': np.mean(mdd),
            'down': np.mean(down)
        }

        return analysis
