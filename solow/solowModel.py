import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rc
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

import solow_cases
from ornstein_uhlenbeck import OrnsteinUhlenbeck

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 16})
plt.rc('font', size=12)  # controls default text sizes
plt.rc('axes', titlesize=12)  # fontsize of the axes title
plt.rc('axes', labelsize=12)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
plt.rc('legend', fontsize=12)  # legend fontsize
plt.rc('figure', titlesize=20)


class SolowModel(object):
    def __init__(self, params: dict,
                 xi_args=None):
        """"""

        if xi_args is None:
            xi_args = dict(decay=0.2, diffusion=1.0)
        self.params = params
        self.xi_args = xi_args

        self.path = None
        self.t_end = 0

        arg_order = [
            "tech0", "rho", "epsilon", "tau_y", "tau_s", "tau_h", "dep",
            "saving0", "gamma", "h_h", "beta1", "beta2", "c1", "c2", "s0"
        ]
        self.args = [params[v] for v in arg_order]

    def solve(self, start: np.ndarray, t_end: float = 1e5,
              case: str = 'general', seed: int = 40,
              save: bool = False) -> pd.DataFrame:
        """

        Parameters
        ----------
        start   :   np.ndarray
        t_end   :   float
        case    :   str
        seed    :   int
        save    :   bool

        Returns
        -------
        path    :   pd.DataFrame

        """
        f = {
            "general": solow_cases.general,
            "direct_feedback": solow_cases.direct_feedback,
            "sentiment_feedback": solow_cases.sentiment_feedback,
            "limit_kd": solow_cases.limit_kd,
            "limit_ks": solow_cases.limit_ks,
            "unbounded_information": solow_cases.unbounded_information,
            "general_no_patch":solow_cases.general_no_patch,
        }[case]

        # Arguments for the IVP solver
        np.random.seed(seed)
        t_eval = np.arange(1, int(t_end) - 1)
        self.t_end = t_end
        ou = OrnsteinUhlenbeck(**self.xi_args)
        args = self.args + [ou]

        # Generate path of variables
        path = solve_ivp(f, t_span=(1, t_end), y0=start, method='RK45',
                         t_eval=t_eval, args=args, atol=1e-5)
        print(path.message)

        cols = ['y', 'ks', 'kd', 's', 'h', 'g', 'saving']
        df = pd.DataFrame(path.y.T, columns=cols)

        if save:
            file = open(self._name_gen(case, 'df'), 'wb')
            pickle.dump(df, file)
            file.close()

        self.path = df

        return df

    def asymptotic_growth(self, case: str = 'general') -> list:
        """ Calculate the asymptotic growth rates given a specific case

        Parameters
        ----------
        case    :   str

        Returns
        -------
        rates   :   list
            list of growth rates for y, ks, kd
        """
        p = self.params

        if case in ['general', 'unbounded_information', 'sentiment_feedback',
                    'limit_kd']:
            temp = p['beta2'] * p['c2'] * p['gamma']
            psi_y = p['epsilon'] / (1 - 0.75 * p['rho'] * temp)
            psi_kd = (0.75 * temp * p['epsilon']) / (1 - 0.75 * p['rho'] * temp)
            return [psi_y, psi_y, psi_kd]

        elif case is 'direct_feedback':
            temp = (1 - p['rho'] * p['gamma'])
            psi_y = p['epsilon'] / temp
            psi_kd = (0.75 * p['c2'] * p['gamma'] * p['epsilon']) / temp
            return [psi_y, psi_y, psi_kd]

        elif case is 'limit_ks':
            psi_y = p['epsilon'] / (1 - p['rho'])
            return [psi_y, psi_y, 0]

        else:
            return [None, None, None]

    def overview(self, case: str = 'general', asymptotics: bool = True):

        plt.rc('text', usetex=False)
        df = self.path
        t = df.index.values

        # Set up the long-term growth rates and sentiment equilibrium
        rates = self.asymptotic_growth(case)

        # Generate a Figure
        fig, ax_lst = plt.subplots(1, 3)
        fig.set_size_inches(18, 6)

        # Production
        if asymptotics:
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
        if asymptotics:
            props = dict(linestyle='--', alpha=0.7)
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
        tgt_max, tgt = self._sentiment_average(case, bound=True)
        if tgt > tgt_max:
            s_bar = [-.5, .5]
        else:
            s_bar = self._sentiment_average(case, bound=False)

        props = dict(linestyle='--', alpha=0.7, color='Orange', linewidth=1.5)
        ax_lst[2].axhline(s_bar[0], **props,
                          label='s_bar {:.2f} & {:.2f}'.format(*s_bar))
        ax_lst[2].axhline(s_bar[1], **props)
        ax_lst[2].plot(df.s, label='Sentiment', color='Blue')
        ax_lst[2].set_title("Sentiment (s)")
        ax_lst[2].set_xlabel("Time")
        ax_lst[2].set_ylabel("Sentiment (s)")
        ax_lst[2].set_xlim(0, df.index[-1])
        ax_lst[2].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax_lst[2].legend()

    def analytics(self, case: str = 'general') -> dict:
        """ Determine the analytical information for the given case, and return
        it in a dictionary form

        Parameters
        ----------
        case    :   str

        Returns
        -------
        analytics   :   dict
        """
        p = self.params

        # Asymptotic growth rates and long-term averages
        rates = self.asymptotic_growth(case)
        s_bar = self._sentiment_average(case, bound=False)

        # Condition for tanh(gamma*psi_y) = gamma*psi_y
        temp = p['rho'] * p['beta2'] * p['c2'] * p['gamma']
        gamma_psi = (p['gamma'] * p['epsilon']) / (1 - 0.75 * temp)

        # Condition for multiple equilibria with s bar
        tgt_max, tgt = self._sentiment_average(case, bound=True)

        res = dict(psi_y=rates[0], psi_ks=rates[1], psi_kd=rates[2],
                   s_bar_top=s_bar[0], s_bar_bottom=s_bar[1],
                   gamma_psi_y=(gamma_psi, gamma_psi < 1),
                   s_bar_cond=(tgt_max, tgt, tgt <= tgt_max))

        return res

    def save_model(self):
        """ Save the model and return the location

        Returns
        -------
        path    :   str
        """
        file = open(self._name_gen(case='MODEL', kind='obj'), 'wb')
        pickle.dump(self, file)
        file.close()

    def _sentiment_average(self, case: str, bound: bool = False):
        """ Determine the values of any conditions that might arise in the
        derivation of the asymptotic growth rates

        Parameters
        ----------
        case    :   str

        Returns
        -------
        cond    :   dict
        """
        p = self.params
        rates = self.asymptotic_growth(case)

        # Function for the solutions to long-term s
        f = lambda x: np.arctanh(x) - p['beta1'] * x
        tgt = p['beta2'] * p['gamma'] * rates[0]

        if bound:
            # Determine the boundary that the target should be below
            temp = minimize(lambda x: -1 * f(x), x0=np.array([-0.5]),
                            bounds=((-0.99, 0),), tol=1e-15)
            return f(temp.x[0]), tgt
        else:
            # Determine intersections and thus long-term averages
            intersects = []
            for x0 in np.linspace(-.999, 0.999, 11):
                s = minimize(lambda x: np.sqrt((f(x) - tgt) ** 2), x0=x0,
                             bounds=((-0.99, 0.99),), tol=1e-15)
                intersects.extend([s.x[0]])
            return [max(intersects), min(intersects)]

    def _name_gen(self, case: str, kind: str = 'df') -> str:
        """ Generate the file name for a given set of parameters

        Parameters
        ----------
        case    :   str
        kind    :   file type formatter

        Returns
        -------
        name    :   str

        """
        p = self.params

        parts = [
            case,
            't{:.0e}'.format(self.t_end),
            'g{:d}'.format(p['gamma']),
            'e{:.1e}'.format(p['epsilon']),
            'c1_{:.1e}'.format(p['c1']),
            'c2_{:.1e}'.format(p['c2']),
            'b1_{:.1f}'.format(p['beta1']),
            'b2_{:.1f}'.format(p['beta2']),
            'ty{:.1e}'.format(p['tau_y']),
            'ts{:.1e}'.format(p['tau_s']),
            'th{:.1e}'.format(p['tau_h']),
            'lam{:.2f}'.format(p['saving0'])
        ]

        name = '_'.join(parts)
        name = 'pickles/' + name + '.' + kind
        return name
