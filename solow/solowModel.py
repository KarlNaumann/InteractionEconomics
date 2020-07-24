import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

from capital_market import CapitalMarket
from firm import Firm
from household import Household
from ornstein_uhlenbeck import OrnsteinUhlenbeck


class SolowModel(object):
    def __init__(self, hh_kwargs: dict, firm_kwargs: dict, capital_kwargs: dict,
                 tech_rate: float, ou_kwargs: dict):
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
        tech_rate   :   float
            technology growth rate


        Returns
        ----------
        SolowModel instance
        """
        # Entities and Variables
        self.tech_rate = tech_rate
        self.firm = Firm(**firm_kwargs)
        self.household = Household(**hh_kwargs)
        self.cm = CapitalMarket(**capital_kwargs)
        self.ou_process = OrnsteinUhlenbeck(**ou_kwargs)

        # Storage
        self.path = None

    def solve(self, initial_values: dict, t0: float = 0, t_end: float = 1e2):
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
            'capital_market': self.cm,
        }

        # Model
        f = self._step

        # Transform the input values
        initial_values = self._initial_values(initial_values)

        # Solve the initial value problem
        path = solve_ivp(f, t_span=(t0, t_end), y0=initial_values, max_step=1.0,
                         method='RK45', t_eval=np.arange(int(t0), int(t_end) - 1),
                         args=(entities, self.ou_process, self.tech_rate))

        self.path = path
        self.vars = self._path_df(path.y, t0, t_end)

        return path

    def _step(self, t, values: list, entities: dict, ou_process: OrnsteinUhlenbeck,
              tech_rate: float):

        # Unpack inputs
        y, ks, kd, s, h, tech, cons, excess, r = values
        # Note: these are the values at period t, and we are going for period t+delta

        v_tech = tech_rate * tech
        news = ou_process.euler_maruyama(t)

        # Determine consumption and investment of hh at time t
        consumption, investment = entities['household'].consumption(y)
        v_cons = consumption - cons

        # Capital level given the supply and demand
        if entities['capital_market'].dyn_demand:
            k = min([ks, np.exp(kd)])
            v_excess = -excess + max([ks - np.exp(kd), 0])
        else:
            k = ks
            v_excess = 0

        # Determine new production level and velocity of production
        factors = {'k': k, 'n': 1, 'tech': tech}
        y_new, k_ret = entities['firm'].production(factors)
        v_y = entities['firm'].production_velocity(curr=y, new=y_new)

        # Effective interest rate earned
        dep = entities['capital_market'].depreciation
        pop = entities['capital_market'].pop_growth
        v_r = -r + (k / ks) * (k_ret - dep) - pop

        # Capital supply ( - depreciation + new investment)
        v_ks = entities['capital_market'].supply_velocity(ks, investment)

        # Determine the velocity of capital demand
        v_ln_y = v_y / y  # entities['firm'].production_velocity(curr=y, new=y_new, ln=True)
        if entities['capital_market'].dyn_demand:
            v_kd, v_s, v_h = entities['capital_market'].demand_velocity(s, h, news, v_ln_y, excess)
        else:
            v_kd, v_s, v_h = [0, 0, 0]

        return [v_y, v_ks, v_kd, v_s, v_h, v_tech, v_cons, v_excess, v_r]

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
        r = (k / given['ks']) * (k_ret - self.cm.depreciation) - self.cm.pop_growth
        initial_values.append(r)
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

    def visualise(self, save: str = ''):
        """ Generate a plot to visualise the outputs"""
        df = self.vars
        fig, ax_lst = plt.subplots(4, 2)
        fig.set_size_inches(15, 10)

        # Production and growth
        ax_lst[0, 0].plot(df.loc[:, 'y'])
        ax_lst[0, 0].set_title('Production')
        ax_lst[0, 1].plot(df.loc[:, 'y'].pct_change())
        ax_lst[0, 1].set_title('Production Growth Rates')

        # Capital Markets
        ax_lst[1, 0].plot(np.exp(df.loc[:, 'kd']), label='Demand')
        ax_lst[1, 0].plot(df.loc[:, 'ks'], label='Supply')
        ax_lst[1, 0].legend()
        ax_lst[1, 0].set_title('Capital Supply and Demand')
        ax_lst[1, 1].plot(df.loc[:, 'r'])
        ax_lst[1, 1].set_title('Effective Interest Rate')

        # Capital Demand Dynamics
        ax_lst[2, 0].plot(df.loc[:, 's'])
        ax_lst[2, 0].set_title('Sentiment')
        ax_lst[2, 1].plot(df.loc[:, 'h'])
        ax_lst[2, 1].set_title('Information')

        plt.tight_layout()

        if save is not '':
            plt.savefig(save, bbox_inches='tight')
        plt.show()
