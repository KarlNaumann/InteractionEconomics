import numpy as np


class Household(object):
    def __init__(self, savings_rate: float):
        """ Household class for the dynamic Solow model. At this point the
        household receives income and saves a proportion of this.

        Parameters
        ----------
        savings_rate   :   float
            constant savings rate

        Attributes
        ----------
        savings_rate       :   float
            constant savings rate
        income_history     :   list of float
            incomes received during the update method
        dividend_history   :   list of float
            dividends received during the update method

        Methods
        -------
        consumption
            function that takes the households income from firm production and
            from dividends and returns the consumption and investment
        get_income
            returns np.array version of incomes list
        get_dividends
            returns the dividends received as np.array
        """

        self.savings_rate = savings_rate
        self.income_history = []
        self.dividend_history = []

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

        total_income = income + dividend
        self.income_history.append(total_income)
        self.dividend_history.append(dividend)

        consumption = (1 - self.savings_rate) * total_income
        investment = self.savings_rate * total_income

        return consumption, investment

    def get_income(self):
        """Function to get incomes as np.array"""
        return np.array(self.income_history)

    def get_dividends(self):
        """Function to get dividends as np.array"""
        return np.array(self.dividend_history)
