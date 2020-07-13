import numpy as np

class Household(object):
    def __init__(self, kappa: float):
        """ Household class for the dynamic Solow model. At this point the
        household receives income and saves a proportion of this.

        Parameters
        ----------
        kappa   :   float
            constant savings rate

        Attributes
        ----------
        kappa       :   float
            constant savings rate
        incomes     :   list of float
            incomes received during the update method
        dividends   :   list of float
            dividends received during the update method

        Methods
        -------
        update
            function that takes the households income from firm production and
            from dividends and returns the consumption and investment
        get_income
            returns np.array version of incomes list
        get_dividends
            returns the dividends received as np.array
        """

        self.kappa = kappa
        self.incomes = []
        self.dividends = []

        def update(self, income: float, dividend: float):
            """

            Parameters
            ----------
            income      :   float
                income from production received
            dividends   :   float
                income from dividends due to firm ownership

            Returns
            -------
            consumption :   float
                income spent on consumption
            investment  :   float
                income spent on savings
            """

            self.incomes.append(income)
            self.dividends.append(dividend)

            consumption = (1-self.kappa) * income
            investment = self.kappa * income

            return consumption, investment

        def get_income(self):
            """Function to get incomes as np.array"""
            return np.array(self.incomes)

        def get_dividends(self):
            """Function to get dividends as np.array"""
            return np.array(self.dividends)