import numpy as np
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
                    gamma:float, c1: float, c2: float, c3: float
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
        self.capital_market = CapitalMarket(**capital_kwargs)
        self.ou_process = OrnsteinUhlenbeck(**ou_kwargs)

        # Storage
        self.path = None

    def solve(self, initial_values: list, model: str, t0: float = 0, t_end: float = 1e2):
        """ Iterate through the Solow model to generate a path for output,
        capital (supply & demand), and sentiment

        Parameters
        ----------
        initial_values  :   list
            initial values in the order: [ production, capital supply,
            capital demand, sentiment, information ]
        model   :   str
            which model to use
        t0
        t_end

        Returns
        -------

        """
        # Agents involved in the economy
        entities = {
            'household': self.household,
            'firm': self.firm,
            'capital_market': self.capital_market,
        }

        # Model
        models = {'draft': self._step_draft, 'excess': self._step_excess}
        try:
            f = models[model]
        except KeyError:
            "model not in selection"

        # Solve the initial value problem
        path = solve_ivp(f, t_span=(t0, t_end), y0=initial_values,
                         method='RK45', t_eval=np.arange(int(t0), int(t_end) - 1),
                         args=(entities, self.ou_process, self.tech_rate))

        self.path = path

        return path

    @staticmethod
    def _step_excess(t, values: list, entities: dict, ou_process: OrnsteinUhlenbeck,
                     tech_rate: float):
        """ Function that defines the order of operations for the solver of the
        system.

        Parameters
        ----------
        t   :   float
            time at which the system is evaluated
        y   :   list
            ordered list of system parameters - [ production, capital supply,
            capital demand, sentiment, information ]
        entities    :   dict
            dictionary of the different classes {'household','firm',
            'capital_market'}
        ou_process  :   OrnsteinUhlenbeck instance
            ornstein uhlenbeck exogenous noise process
        tech_rate   :   float
            technology growth rate

        Returns
        -------
        x   :   list
            ordered list of the updated parameters
        """
        # Extract parameters
        y, ks, kd, s, h, tech, excess, cons = values

        v_tech = tech_rate * tech
        news = ou_process.euler_maruyama(t)

        # Determine the capital level and excess
        if entities['capital_market'].dyn_demand:
            capital = min([ks, kd])
            v_excess = max([ks - kd, 0]) - excess
        else:
            capital = ks
            v_excess = 0
            excess = 0

        # Determine new production level and velocity of production
        y_new = entities['firm'].production({'k': capital, 'n': 1, 'tech': tech})
        v_y = entities['firm'].production_velocity(curr_prod=y, new_prod=y_new)

        # Household decision
        consumption, investment = entities['household'].consumption(y)
        v_cons = consumption + excess - cons

        # Determine the velocity of capital supply
        v_ks = entities['capital_market'].supply_velocity(capital, investment)
        v_ks -= excess

        # Convert production velocity to velocity of log
        v_log_y = v_y / y

        # Determine the velocity of capital demand
        if entities['capital_market'].dyn_demand:
            v_kd, v_s, v_h = entities['capital_market'].demand_velocity(s, h, news, v_log_y)
        else:
            v_kd, v_s, v_h = [0, 0, 0]

        # Convert velocity of log capital demand to regular capital demand
        v_kd *= kd

        # [production, capital supply, capital demand, sentiment, information, tech]
        return [v_y, v_ks, v_kd, v_s, v_h, v_tech, v_excess, v_cons]

    def _step_draft(self, t, values: list, entities: dict, ou_process: OrnsteinUhlenbeck,
                    tech_rate: float):

        # Unpack inputs
        y, ks, kd, s, h, tech, cons = values

        v_tech = tech_rate * tech
        news = ou_process.euler_maruyama(t)

        # Capital level
        if entities['capital_market'].dyn_demand:
            k = min([ks, kd])
        else:
            k = ks

        # Determine new production level and velocity of production
        y_new = entities['firm'].production({'k': k, 'n': 1, 'tech': tech})
        v_y = entities['firm'].production_velocity(curr_prod=y, new_prod=y_new)

        # Determine consumption and investment
        consumption, investment = entities['household'].consumption(y_new)
        v_cons = consumption - cons

        # Capital supply
        v_ks = entities['capital_market'].supply_velocity(ks, investment)

        # Determine the velocity of capital demand
        v_ln_y = v_y / y_new
        if entities['capital_market'].dyn_demand:
            v_ln_kd, v_s, v_h = entities['capital_market'].demand_velocity(s, h, news, v_ln_y)
        else:
            v_kd, v_s, v_h = [0, 0, 0]

        v_kd = kd * v_ln_kd

        return [v_y, v_ks, v_kd, v_s, v_h, v_tech, v_cons]

    def visualise(self):

        data = self.path.y

        fig, ax_lst = plt.subplots(3, 2)
        fig.set_size_inches(20, 15)
        # Production and Tech
        ax_lst[0, 0].plot(self.path.t, data[0, :],label='Production')
        ax_lst[0, 0].set_title('Production')
        ax_lst[0, 0].plot(self.path.t, data[-1],label='Consumption')
        ax_lst[0, 0].legend()
        # Capital Markets
        ax_lst[1, 0].plot(self.path.t, data[1, :], label='Supply')
        ax_lst[1, 0].plot(self.path.t, data[2, :], label='Demand')
        ax_lst[1, 0].legend()
        ax_lst[1, 0].set_title('Capital Markets')
        ax_lst[1, 1].plot(self.path.t, np.minimum(data[1, :], data[2, :]))
        ax_lst[1, 1].set_title('Capital Applied')
        # Capital Demand Dynamics
        ax_lst[2, 0].plot(self.path.t, data[3, :])
        ax_lst[2, 0].set_title('Sentiment')
        ax_lst[2, 1].plot(self.path.t, data[4, :])
        ax_lst[2, 1].set_title('Information')
        plt.show()
