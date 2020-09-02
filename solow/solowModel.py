import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numdifftools import Jacobian
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from ornstein_uhlenbeck import OrnsteinUhlenbeck


class SolowModel(object):
    def __init__(self, y_args: dict, cap_args: dict, theta: float = 0.2,
                 sigma: float = 1.0):
        """ Class implementing the Solow model

        Parameters
        ----------
        y_args  :   dict
            (optional) dictionary of production arguments (incl. tech0, rho, e,
            tau_y, dep)
        cap_args    :   dict
            (optional) dictionary of capital markets arguments (incl. saving,
            tau_h, tau_s, c1, c2, beta1, beta2, gamma, h_h)
        theta, sigma    :   float
            decay and variance of the Ornstein-Uhlenbeck process


        Returns
        ----------
        SolowModel instance
        """
        # Entities and Variables
        self.y_args = y_args
        self.cap_args = cap_args
        self.ou = OrnsteinUhlenbeck(decay=theta, diffusion=sigma)

        # Storage
        self.path = None
        self.recessions = None

        # Properties
        self.case_list = ['general', 'general_r', 'limit_ks', 'limit_kd']

    def overview(self, start: list, y_args: dict, cap_args: dict, gamma: float,
                 h_k: float = 1e1, h_h: float = 1e1, theta: float = 0.2,
                 sigma: float = 1, case: str = 'general', t_end: float = 1e5,
                 save: str = '') -> pd.DataFrame:

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
        h_k, h_h   :   float
            Extent to which tanh will approximate the min() for k and for gamma
        theta,sigma :   float
            Decay and diffusion of the Ornstein Uhlenbeck process
        case        :   str
            choice of case, can be 'general', 'limit_ks', 'limit_kd'
        t_end       :   float
        save        :   str

        """

        # Arguments
        t_eval = np.arange(1, int(t_end) - 1)
        ou = OrnsteinUhlenbeck(theta, sigma, 0)
        # assert case in self.case_list, "Case not found"
        args = (y_args, cap_args, gamma, h_k, h_h, ou, case)
        start.append(cap_args['saving'])

        # Generate path of variables
        path = solve_ivp(self._step, t_span=(1, t_end), y0=start, method='RK45',
                         t_eval=t_eval, args=args, atol=1e-5)
        print(path.message)
        df = pd.DataFrame(path.y.T,
                          columns=['y', 'ks', 'kd', 's', 'h', 'g', 'saving'])

        k = df.kd - (df.kd - df.ks) * 0.5 * (1 + np.tanh(h_k * (df.kd - df.ks)))
        e = pd.Series(np.arange(df.index[0], df.index[-1])) * y_args['e']
        z = y_args['rho'] * k + e - df.y

        rates = self.long_term_growth(y_args, cap_args, gamma)
        tgt = cap_args['beta2'] * gamma * rates[0]
        long_term_s = self.s_bar(tgt, cap_args['beta1'])

        # Generate a Figure
        fig, ax_lst = plt.subplots(1, 3)
        fig.set_size_inches(12, 4)
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        t = df.index.values

        # Production
        ax_lst[0].plot(t, df.y.iloc[0] + t * rates[0], linestyle='--',
                       color='Orange', label='Asymp. Growth')
        ax_lst[0].plot(df.y, color='Blue', label='Production')
        ax_lst[0].set_title("Log Production (y)")
        ax_lst[0].set_xlabel("Time")
        ax_lst[0].set_ylabel("Log Production")
        ax_lst[0].set_xlim(0, t[-1])
        ax_lst[0].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax_lst[0].legend()

        # Capital Markets
        props = {'linestyle': '--', 'alpha': 0.7}
        ax_lst[1].plot(t, df.ks.iloc[0] + rates[1] * t, label='Ks Asymp.',
                       color='Blue', **props)
        ax_lst[1].plot(t, df.kd.iloc[0] + rates[2] * t, label='Kd Asymp.',
                       color='Orange', **props)
        ax_lst[1].plot(df.ks, label='Ks', color='Blue')
        ax_lst[1].plot(df.kd, label='Kd', color='Orange')
        ax_lst[1].set_title("Capital Markets (ks, kd)")
        ax_lst[1].set_xlabel("Time")
        ax_lst[1].set_ylabel("Log Capital")
        ax_lst[1].set_xlim(0, df.index[-1])
        ax_lst[1].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax_lst[1].legend()

        # Sentiment
        props = {'linestyle': '--', 'alpha': 0.7, 'color': 'Orange'}
        ax_lst[2].axhline(long_term_s[0], **props,
                          label='s_bar {:.4f} & {:.4f}'.format(*long_term_s))
        ax_lst[2].axhline(long_term_s[1], **props)
        ax_lst[2].plot(df.s, label='Sentiment', color='Blue')
        ax_lst[2].set_title("Sentiment (s)")
        ax_lst[2].set_xlabel("Time")
        ax_lst[2].set_ylabel("Sentiment (s)")
        ax_lst[2].set_xlim(0, df.index[-1])
        ax_lst[2].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax_lst[2].legend()

        """
        # Feedback strength
        ax_lst[1, 1].plot(df.g)
        ax_lst[1, 1].set_title("Feedback Strength Multiplier")
        ax_lst[1, 1].set_xlabel("Time")
        ax_lst[1, 1].set_ylabel("Multiplier")
        ax_lst[1, 1].set_xlim(0, df.index[-1])
        ax_lst[1, 1].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

        # Limit cycles in z
        ax_lst[0, 2].plot(df.s, z, alpha=0.5)
        ax_lst[0, 2].set_title("Z")
        ax_lst[0, 2].set_xlabel("Sentiment")
        ax_lst[0, 2].set_ylabel("Z")
        ax_lst[0, 2].set_xlim(-1, 1)

        # Starting point information
        texts = ['y0: {:.1f}, '.format(start[0]),
                 's0: {:.1f}, h0: {:.1f}'.format(start[3], start[4]),
                 'ks0: {:.1f}, kd0: {:.1f}'.format(start[1], start[2]),
                 'gamma:{:.1e}'.format(gamma),
                 'h_h: {:.1e}'.format(h_h)]
        ax_lst[1, 2].text(0.5, 0.5, '\n'.join(texts),
                          transform=ax_lst[1, 2].transAxes, fontsize=14,
                          verticalalignment='center',
                          horizontalalignment='center', bbox=props)
        ax_lst[1, 2].set_axis_off()
        """

        plt.tight_layout()
        if save is not '':
            plt.savefig(save, bbox_inches='tight')
        plt.show()

        return df

    def solve(self, start: list, t0: float = 0, t_end: float = 1e2,
              stoch: bool = True, theta: float = 0.2, sigma: float = 1.0,
              y_args: dict = None, cap_args: dict = None,
              case: str = 'general') -> pd.DataFrame:
        """ Iterate through the Solow model to generate a path for output,
        capital (supply & demand), and sentiment

        Parameters
        ----------
        start       :   list
            initial values, order = y, ks, kd, s, h, r
        t0, t_end   :   float
            integration interval
        stoch       :   bool
            whether to solve the system with a stochastic parameter
        theta, sigma:   float
            decay and variance of the Ornstein-Uhlenbeck process
        y_args      :   dict
            (optional) dictionary of production arguments (incl. tech0, rho, e,
            tau_y, dep)
        cap_args    :   dict
            (optional) dictionary of capital markets arguments (incl. saving,
            tau_h, tau_s, c1, c2, beta1, beta2, gamma, h_h)
        case        :   str
            choice of case, can be 'general', 'limit_ks', 'limit_kd'

        Returns
        -------

        """

        # Arguments
        t_eval = np.arange(int(t0), int(t_end) - 1)
        if cap_args is None:
            cap_args = self.cap_args
        if y_args is None:
            y_args = self.y_args

        if not stoch:
            ou = None
        elif y_args is None or cap_args is None:
            ou = self.ou
        else:
            ou = OrnsteinUhlenbeck(decay=theta, diffusion=sigma)

        assert case in self.case_list, "Case not found"
        gamma, h_h = self.cap_args['gamma'], self.cap_args['h_h']
        args = (y_args, cap_args, gamma, 10, h_h, ou, case)

        # Generate path of variables
        path = solve_ivp(self._step, t_span=(t0, t_end), y0=start,
                         method='RK45', t_eval=t_eval, args=args)
        print(path.message)
        df = pd.DataFrame(path.y.T, columns=['y', 'ks', 'kd', 's', 'h', 'g'])

        self.path = df

        return df

    def _step(self, t, values: list, y_args: dict, cap_args: dict,
              gamma: float, h_k: float = 10, h_h: float = 10,
              ou_process=None, case: str = 'general') -> list:
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
            strength of the feedback effect
        h_k, h_h    :   float
            Extent to which tanh will approximate the min() for k and for gamma
        ou_process  :   OrnsteinUhlenbeck
        case        :   str
            limiting cases can be chosen in this manner

        Returns
        -------
        velocities  :   list
        """
        y, ks, kd, s, h, gamma_mult, saving = values

        if ou_process is not None:
            news = ou_process.euler_maruyama(t)
        else:
            news = 0

        # Market Clearing
        if case is 'limit_kd':
            k = kd
        elif case is 'limit_ks':
            k = ks
        else:
            k = kd - (kd - ks) * 0.5 * (1 + np.tanh(h_k * (kd - ks)))

        # Dynamic Saving
        if case is 'dynamic_saving':
            v_saving = -saving + (k / ks) * cap_args['saving']
        else:
            v_saving = 0

        # Production changes
        z = y_args['rho'] * k + y_args['e'] * t - y
        v_y = (np.log(y_args['tech0']) * np.exp(z) - 1) / y_args['tau']

        # Capital supply changes
        v_ks = (saving * np.exp(y - ks) - y_args['dep'] * np.exp(k - ks))

        # Gamma multiplier only if there is no limiting kd case
        if case is 'limit_kd':
            gamma_mult_n = 1
        else:
            gamma_mult_n = 0.5 * (1 + np.tanh(h_h * (ks - kd)))

        v_g_s = gamma_mult_n - gamma_mult

        # Information velocity
        if case is 'tau_h0':
            v_h = 0
            h = np.tanh(gamma * gamma_mult_n * v_y + news)
        else:
            v_h = -h + np.tanh(gamma * gamma_mult_n * v_y + news)
            v_h = v_h / cap_args['tau_h']

        # Sentiment Velocity
        force_s = cap_args['beta1'] * s + cap_args['beta2'] * h
        if case is 'tau_s0':
            v_s = -s + np.tanh(force_s)
            s = np.tanh(force_s)
        else:
            v_s = (-s + np.tanh(force_s)) / cap_args['tau_s']

        # Capital demand velocity
        if case is 'direct_feedback':
            v_h = -h + np.tanh(news)
            force_s = cap_args['beta1'] * s + cap_args['beta2'] * h
            v_s = -s + np.tanh(force_s + gamma * gamma_mult_n * v_y)
            v_kd = sum([
                cap_args['c1'] * v_s,
                cap_args['c2'] * (s - cap_args['s0'])
            ])
        else:
            v_kd = cap_args['c1'] * v_s + cap_args['c2'] * (s - cap_args['s0'])

        return [v_y, v_ks, v_kd, v_s, v_h, v_g_s, v_saving]

    def long_term_growth(self, y_args: dict, cap_args: dict,
                         gamma: float) -> list:

        temp = cap_args['beta2'] * cap_args['c2'] * gamma

        # Long-term rates
        psi_y = y_args['e'] / (1 - 0.75 * y_args['rho'] * temp)
        psi_ks = psi_y
        psi_kd = (temp * y_args['e']) / (1 - 0.75 * y_args['rho'] * temp)
        return [psi_y, psi_ks, psi_kd]

    def s_bar(self, tgt: float, b1: float = 1.1, graph: bool = False) -> list:

        def f(x): return 0.5 * np.log((1 + x) / (1 - x)) - b1 * x

        g = lambda x: (f(x) - tgt) ** 2

        s_plus = minimize(g, x0=np.array([0.5]), bounds=((-0.99, 0.99),),
                          tol=1e-15)
        s_minus = minimize(g, x0=np.array([-0.5]), bounds=((-0.99, 0.99),),
                           tol=1e-15)

        if graph:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlabel('s')
            ax.set_ylabel('f(s)')
            ax.set_title('Graphical Solution to Long-term Sentiment Equilibria')
            # Axes to the middle
            ax.axvline(0, color='gray', alpha=0.7)
            ax.axhline(0, color='gray', alpha=0.7)
            ax.axhline(tgt, label='Tgt at {:.3f}'.format(tgt), color='red')

            x = np.linspace(-1, 1, 101)
            ax.plot(x, f(x))
            ax.scatter(s_plus.x[0], 0,
                       label='Positive Eq. at s={:.5f}'.format(s_plus.x[0]),
                       marker='x', color='black')
            ax.scatter(s_minus.x[0], 0,
                       label='Negative Eq. at s={:.5f}'.format(s_minus.x[0]),
                       marker='x', color='black')
            ax.set_xlim(-1, 1)
            ax.legend()
            plt.tight_layout()
            plt.show()

        return [s_plus.x[0], s_minus.x[0]]

    def recession_timing(self, gdp: pd.Series = None,
                         timescale: float = 63) -> pd.DataFrame:
        """ Calculate the start and end of recessions on the basis of gdp growth.
        Two consecutive periods of length t that have negative growth start a
        recession. Two with positive growth end it.

        Parameters
        ----------
        gdp :   pd.Series
            Time series of production
        timescale   :   float (default 63)
            Timescale in business days over which to evaluate recession, default
            is quarterly (63 business days)

        Returns
        -------
        df  :   pd.DataFrame
            DataFrame containing full cycles in sample. I.e. start at first
            moment of expansion, end at last moment of expansions. Gives the
            expa

        """
        if gdp is None:
            assert self.path is not None, "No GDP saved"
            print("Using saved GDP")
            gdp = self.path.y

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
        try:
            se = [(0, min(recessions))]
        except ValueError:
            return pd.DataFrame(columns=['expansion', 'recession'])

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

        self.recessions = pd.DataFrame(se, columns=['expansion', 'recession'])
        return self.recessions

    def recession_analysis(self, rec: pd.DataFrame = None,
                           gdp: pd.Series = None) -> pd.DataFrame:
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
        if gdp is None:
            assert self.path is not None, "No GDP saved"
            gdp = self.path.y

        if rec is None:
            if self.recessions is None:
                print("Determining Recessions...")
                self.recession_timing(gdp)
            rec = self.recessions

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

        return pd.DataFrame.from_dict(analysis, orient='index')

    """ WORKS IN PROGRESS """

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
            v = self._step(1, point, *args)
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

    def visualise(self, save: str = '', case: str = 'general'):
        """ Generate a plot to visualise the outputs"""
        df = self.path.copy(deep=True)
        rec = self.recession_timing(df.y)
        rec = pd.DataFrame(np.zeros((2, 2)))

        def rec_plot(ax, rec):
            for j in range(rec.shape[0] - 1):
                ax.axvspan(xmin=rec.iloc[j, 1], xmax=rec.iloc[j + 1, 0],
                           color='gainsboro')

        if case in ['general', 'limit_kd']:
            fig, ax_lst = plt.subplots(2, 2)
            fig.set_size_inches(15, 10)
            rec = self.recession_timing(df.y)

            # Production and growth
            ax_lst[0, 0].plot(df.loc[:, 'y'])
            ax_lst[0, 0].set_title('Production')
            rec_plot(ax_lst[0, 0], rec)
            ax_lst[0, 1].plot(df.loc[:, 'y'].pct_change())
            ax_lst[0, 1].set_title('Production Growth Rates')
            rec_plot(ax_lst[0, 1], rec)

            # Capital Markets
            ax_lst[1, 0].plot(df.kd, label='Demand')
            ax_lst[1, 0].set_title('Capital Markets')
            if case != 'limit_kd':
                ax_lst[1, 0].plot(df.ks, label='Supply')
            ax_lst[1, 0].legend()
            rec_plot(ax_lst[1, 0], rec)

            # Capital Demand Dynamics
            ax_lst[1, 1].plot(df.s)
            ax_lst[1, 1].set_title('Sentiment')
            rec_plot(ax_lst[1, 1], rec)

        elif case == 'limit_ks':
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
