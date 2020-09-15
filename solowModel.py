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
        self.sbars = None
        self.t_end = 0

        arg_order = [
            "tech0", "rho", "epsilon", "tau_y", "tau_s", "tau_h", "dep",
            "saving0", "gamma", "h_h", "beta1", "beta2", "c1", "c2", "s0"
        ]
        self.args = [params[v] for v in arg_order]

    def solve(self, start: np.ndarray, t_end: float = 1e5,
              case: str = 'general', seed: int = 40,
              save: bool = False, folder: str = 'pickles/') -> pd.DataFrame:
        """

        Parameters
        ----------
        start   :   np.ndarray
        t_end   :   float
        case    :   str
        seed    :   int
        save    :   bool
        folder  :   str

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
            "general_no_patch": solow_cases.general_no_patch,
        }[case]

        # Arguments for the IVP solver
        np.random.seed(seed)
        self.seed = seed
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
            df.to_csv(self._name_gen(case, 'csv', folder=folder))

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

    def save_model(self, folder):
        """ Save the model and return the location

        Returns
        -------
        path    :   str
        """
        file = open(self._name_gen(case='MODEL', kind='obj', folder=folder),
                    'wb')
        pickle.dump(self, file)
        file.close()

    def load_path(self, csv_filename):
        """ Load a Solow Model from a csv file that contains the integrated path

        Parameters
        ----------
        csv_filename    :   str

        Returns
        -------
        """

        self.path = pd.read_csv(csv_filename)

        temp = csv_filename.split('_')
        self.t_end = float(temp[1][1:])
        self.params = dict(gamma=float(temp[2][1:]), epsilon=float(temp[3][1:]),
                           c1=float(temp[5]), c2=float(temp[7]),
                           beta1=float(temp[9]), beta2=float(temp[11]),
                           tau_s=int(temp[12][2:]), tau_y=int(temp[13][2:]),
                           tau_h=int(temp[14][2:]), saving0=float(temp[15][3:]),
                           tech0=float(temp[16][3:]), dep=float(temp[17][4:]),
                           rho=float(temp[18][3:]), s0=0, h_h=10)

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

        def intersect_kd(s):
            t1 = p['beta2'] * p['gamma'] * p['rho'] * p['c2'] * s
            t2 = p['beta2'] * p['gamma'] * p['epsilon']
            return np.abs(np.arctanh(s) - p['beta1'] * s - np.tanh(t1 + t2))

        intersects = []
        # Search through four quadrants
        quads = [(-0.99, -0.5), (-0.5, 0), (0, 0.5), (0.5, 0.99)]
        for bnds in quads:
            temp = minimize(intersect_kd, 0.5 * sum(bnds), bounds=(bnds,),
                            tol=1e-15)
            intersects.append(temp.x[0])

        self.sbars = [max(intersects), min(intersects)]
        return [max(intersects), min(intersects)]

    def _name_gen(self, case: str, kind: str = 'df',
                  folder: str = 'pickles/') -> str:
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
            't{:07.0e}'.format(self.t_end),
            'g{:07.0f}'.format(p['gamma']),
            'e{:07.1e}'.format(p['epsilon']),
            'c1_{:03.1f}'.format(p['c1']),
            'c2_{:07.1e}'.format(p['c2']),
            'b1_{:03.1f}'.format(p['beta1']),
            'b2_{:03.1f}'.format(p['beta2']),
            'ty{:03.0f}'.format(p['tau_y']),
            'ts{:03.0f}'.format(p['tau_s']),
            'th{:02.0f}'.format(p['tau_h']),
            'lam{:01.2f}'.format(p['saving0']),
            'dep{:07.1e}'.format(p['dep']),
            'tech{:04.2f}'.format(p['tech0']),
            'rho{:04.2f}'.format(p['rho']),
            'seed{:02.0f}'.format(self.seed)
        ]

        name = '_'.join(parts)
        name = folder + name + '.' + kind
        return name
