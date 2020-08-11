import numpy as np


class CapitalMarket(object):
    def __init__(self, static: bool = True, depreciation: float = 0.0002,
                 pop_growth: float = 0, dynamic_kwargs: dict = {},
                 v_excess: bool = True):
        """ Household class for the dynamic Solow model. At this point the
        household receives income and saves a proportion of this.

        Parameters
        ----------
        static          :   float
            whether to use the standard Solow (K = K_s) or to use the dynamic
            capital demand version
        depreciation    :   float
            Rate of depreciation for the capital that is invested
        pop_growth      :   float
            population growth rate
        dynamic_kwargs  :   dict
            parameters to pass to the dynamic capital demand system. Include:
            { tau_s: float, beta1: float, beta2: float, omega_h: float,
                gamma: float }
        v_excess    :   bool
            Whether to calculate the velocity of the excess

        Attributes
        ----------
        v_ks    :   list
            velocity of capital supply over time
        v_kd    :   list
            velocity of capital demand over time
        v_s     :   list
            velocity fo the sentiment over time
        v_h     :   list
            velocity of the information over time

        Public Methods
        --------------
        demand_velocity()
            differential equation for the capital demand

        supply_velocity()
            differential equation for the capital supply
        """

        self.depreciation = depreciation
        self.pop_growth = pop_growth
        self.dyn_demand = not static
        self.v_excess = v_excess
        self.d_kwargs = dynamic_kwargs

    def v_demand(self, s: float, h: float, news: float, d_production: float,
                 excess: float = 0):
        """ Velocity of capital demand. If the demand is static (basic Solow)
        then we update the demand by the same interval as the supply to preserve
        the K_s = K_d relationship. If the demand is dynamic, we apply the dynamic
        sentiment system of Gusev et al. (2015)

        Parameters
        ----------
        s   :   float
            Level of sentiment
        h   :   float
            Level of information
        news    :   float
            Exogenous news
        d_production    :   float
            velocity of log production
        excess  :   float
            excess capital

        Returns
        -------
        v_s     :   float
            Change in the sentiment level
        v_h     :   float
            Change in the information level
        v_kd    :   float
            Change in the log capital demand
        """
        if not self.dyn_demand:
            return 0, 0, 0
        else:
            # Sentiment process
            force_s = self.d_kwargs['beta1'] * s + self.d_kwargs['beta2'] * h
            v_s = (-s + np.tanh(force_s)) / self.d_kwargs['tau_s']
            # Information process
            force_h = sum([(self.d_kwargs['gamma'] * d_production),
                           self.d_kwargs['phi'] * excess,
                           news])
            v_h = (-h + np.tanh(force_h)) / self.d_kwargs['tau_h']
            # Velocity of log capital demand
            v_kd = sum([self.d_kwargs['c1'] * v_s,
                        self.d_kwargs['c2'] * s,
                        self.d_kwargs['c3']])
            return v_kd, v_s, v_h

    def v_supply(self, capital: float, investment: float):
        """ Differential function for the change in capital supply

        Parameters
        ----------
        capital     :   float
            level of capital currently available
        investment  :   float
            level of investment by households

        Returns
        -------
        v_ks    :   change in the supply of capital
        """
        return investment - (self.depreciation + self.pop_growth) * capital

    def clearing(self, ks: float, kd: float, excess: float):
        """ Function for the market clearing of capital

        Parameters
        ----------
        ks  :   float
            Capital supply
        kd  :
            Capital demand
        excess  :   float
            Previous excess capital demand

        Returns
        -------
        k   :   float
            capital used in production
        v_excess  :   float
            velocity of the excess capital
        """

        if self.v_excess:
            return min([ks, kd]), -excess + max([np.log(ks) - kd, 0])
        else:
            return min([ks, kd]), 0


    def eff_earnings(self, k: float, ks: float, k_ret: float):
        """ Effective earnings on capital

        Parameters
        ----------
        k   :   float
            Level of capital used
        ks  :   float
            Level of capital supply i.e. money in bank
        k_ret   :   float
            Return to capital from investment in production

        Returns
        -------

        """
        return (k / ks) * (k_ret - self.depreciation) - self.pop_growth
