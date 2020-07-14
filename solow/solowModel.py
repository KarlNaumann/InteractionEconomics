"""Working file for the draft paper "A Simple Economic Model with Interactions"
by Gusev and Kroujiline on 07/04/2020"""

from firm import Firm
from household import Household
from technology import TechProcess

class SolowModel(object):
    def __init__(self, rho: float, tau_y: float, epsilon: float, delta: float, lam: float):
        """ Class implementing the dynamic Solow model

        Parameters
        ----------
        rho	    : float
                    the capital share of production
        tau_y 	: float
                    the characteristic timescale of output
        epsilon : float
                    the technology growth constant
        delta 	: float
                    the depreciation rate
        lam 	: float
                    the savings rate

        Returns
        ----------
        SolowModel instance
        """

        self.alpha = alpha
        self.tau_y = tau_y
        self.epsilon = epsilon
        self.delta = delta
        self.lam = lam
        self.args = (alpha, tau_y, epsilon, lam, delta)

    def _sentiment(self):
        pass

    def _capital(self):
        pass



if __name__ == "__main__":
    args = {
        'alpha': 0.5,
        'tau_y': 1e3,
        'epsilon': 1e-5,
        'delta': 0.5,
        'lam': 0.5}

    sm = SolowModel(**args)
