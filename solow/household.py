import numpy as np


class Household(object):
    def __init__(self, savings_rate: float, static: bool = True,
                 dynamic_kwargs: dict = {}):
        """ Household class for the dynamic Solow model. At this point the
        household receives income and saves a proportion of this.

        Parameters
        ----------
        savings_rate    :   float
            constant savings rate
        static          :   boolean
            whether to use the standard (constant savings rate) or dynamic
            (mean-field index) consumption decision process
        dynamic_kwargs  :   dict
            optional keywords for the dynamic consumption case. Must include:
            {max_rate: float, min_rate: float, theta:float, reference:float}

        Attributes
        ----------
        savings_rate       :   float
            constant savings rate
        income_history     :   list of float
            total income received across time
        dividend_history   :   list of float
            dividends received across time
        saving_rate_history:   list of float
            savings rate over time

        Public Methods
        --------------
        consumption
            function that takes the households income from firm production and
            from dividends and returns the consumption and investment
        get_income
            returns np.array version of incomes list
        get_dividends
            returns the dividends received as np.array
        """

        self.savings_rate = savings_rate
        self.static = static
        self.income_history = []
        self.dividend_history = []
        self.saving_rate_history = []
        self.dynamic_kwargs = dynamic_kwargs

    def consumption(self, income: float, dividend: float):
        """

        Parameters
        ----------
        income      :   float
            income from production received
        dividend    :   float
            income from dividends due to firm ownership

        Returns
        -------
        consumption :   float
            income spent on consumption
        investment  :   float
            income spent on savings
        """

        tot_income = income + dividend
        self.income_history.append(tot_income)
        self.dividend_history.append(dividend)

        if self.static:
            self.saving_rate_history.append(self.savings_rate)
            return self._static_consumption(tot_income)
        else:
            return self._dynamic_consumption(tot_income, self.income_history[-2])

    def _static_consumption(self, total_income: float):
        """ Apply the standard constant savings rate approach

        Parameters
        ----------
        total_income:   float
            total income for the period
        Returns
        -------
        consumption :   float
            income spent on consumption
        investment  :   float
            income spent on savings

        """
        consumption = (1 - self.savings_rate) * total_income
        investment = self.savings_rate * total_income
        return consumption, investment

    def _dynamic_consumption(self, total_income: float, prior_income: float):
        """ PLACEHOLDER FOR DYNAMIC CASE """

        # savings rate based on the s-shaped function
        self.savings_rate = self._shifted_logistic(prior_income,
                                                   **self.dynamic_kwargs)
        self.saving_rate_history.append(self.savings_rate)

        consumption = (1 - self.savings_rate) * total_income
        investment = self.savings_rate * total_income
        return consumption, investment

    @staticmethod
    def _shifted_logistic(x: float, max_rate: float, min_rate: float,
                          theta: float, reference: float):
        """ Shifted logistic function used by households to adapt their savings
        rate based on the prior period's consumption

        Parameters
        ----------
        x           :   float
            prior period observation by the household of other households
        max_rate    :   float
            maximum savings rate possible e.g. via golden-rule
        min_rate    :   float
            minimum savings rate, must be >0
        theta       :   float
            "Temperature", that determines the spread across which there is
            variety in the savings rate (slope of shift)
        reference   :   float
            Reference value to which x is compared

        Returns
        -------
        new_rate    :   float
            New savings rate determined by the system
        """

        strength = 1 + np.exp(2 * theta * (reference - x))
        new_rate = min_rate + (max_rate - min_rate) / strength

        return new_rate

    def get_income(self):
        """Function to get incomes as np.array"""
        return np.array(self.income_history)

    def get_dividends(self):
        """Function to get dividends as np.array"""
        return np.array(self.dividend_history)
