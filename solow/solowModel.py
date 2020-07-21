import numpy as np
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

    def solve(self, initial_values: list, t0: float = 0, t_end: float = 1e2):
        """ Iterate through the Solow model to generate a path for output,
        capital (supply & demand), and sentiment

        Parameters
        ----------
        initial_values  :   list
            initial values in the order: [ production, capital supply,
            capital demand, sentiment, information ]
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

        path = solve_ivp(self._step, t_span=(t0, t_end), y0=initial_values,
                         method='RK45', t_eval=np.arange(int(t0), int(t_end) - 1),
                         args=(entities, self.ou_process, self.tech_rate))
        
        """
        t = np.linspace(t0, t_end, 5 * (int(t_end) - int(t0)))
        path = np.zeros((t.shape[0], len(initial_values)))
        path[0, :] = initial_values
        for i, t in enumerate(np.arange(int(t0), int(t_end))):
            delta = self._step(t, list(path[i - 1, :]), entities, self.ou_process,
                               self.tech_rate)
            path[i, :] = path[i - 1, :] + delta
        """

        self.path = path

        return path

    @staticmethod
    def _step(t, values: list, entities: dict, ou_process: OrnsteinUhlenbeck,
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
        production, ks, kd, sentiment, information, tech = values
        v_tech = tech_rate * tech

        # Determine the capital level and excess
        if entities['capital_market'].dyn_demand:
            capital = min([ks, kd])
            excess = max([ks - kd, 0])
        else:
            capital = ks
            excess = 0

        # Determine new production level and velocity of production
        y = entities['firm'].production({'k': capital, 'n': 1, 'tech': tech})
        v_y = entities['firm'].production_velocity(curr_prod=production,
                                                   new_prod=y)

        # Household decision
        consumption, investment = entities['household'].consumption(y, excess)
        # Determine the velocity of capital supply
        v_ks = entities['capital_market'].supply_velocity(capital, investment)

        # Convert production velocity to velocity of log
        d_production = v_y / y

        # Determine the velocity of capital demand
        if entities['capital_market'].dyn_demand:
            news = ou_process.euler_maruyama(t)
            v_kd, v_s, v_h = entities['capital_market'].demand_velocity(sentiment,
                                                                        information,
                                                                        news,
                                                                        d_production)
        else:
            v_kd, v_s, v_h = [0, 0, 0]

        # Convert velocity of log capital demand to regular capital demand
        v_kd *= kd

        # [production, capital supply, capital demand, sentiment, information, tech]
        return [v_y, v_ks, v_kd, v_s, v_h, v_tech]


if __name__ == "__main__":
    hh_kwargs = {'savings_rate': 0.3, 'static': True}
    firm_kwargs = {'prod_func': 'cobb-douglas', 'parameters': {'rho': 1 / 3}}
    capital_kwargs = {
        'static': True, 'depreciation': 0.2, 'pop_growth': 0.005,
        'dynamic_kwargs': {
            'tau_s': 1 / 0.004, 'beta1': 1.1, 'beta2': 1.0, 'tau_h': 1 / 0.04,
            'gamma': 2000, 'c1': 1, 'c2': 0.00015, 'c3': 0
        }
    }
    ou_kwargs = {'decay': -0.2, 'drift': 0, 'diffusion': 2.5, 't0': 0}

    sm = SolowModel(hh_kwargs=hh_kwargs,
                    firm_kwargs=firm_kwargs,
                    capital_kwargs=capital_kwargs,
                    tech_rate=0.00005,
                    ou_kwargs=ou_kwargs)
    # [production, capital supply, capital demand, sentiment, information]
    initial_values = [1, 1, 1, 1, 1]
    sm.solve()
