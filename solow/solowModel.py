"""Working file for the draft paper "A Simple Economic Model with Interactions"
by Gusev and Kroujiline on 07/04/2020"""

from matplotlib.ticker import ScalarFormatter

xfmt = ScalarFormatter()
xfmt.set_powerlimits((-15, 15))


class SolowModel(object):
    def __init__(self, alpha: float, tau_y: float, epsilon: float, delta: float, lam: float):
        """Numerical approach to the second order differential in eq. 5
        requires rewriting via x[0] = K, x[1] = K', thus converting the
        problem into a first-order differential equation

        Parameters
        ----------
        alpha	: float
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

    def classicApproximateSolution(self):
        pass

    def classicSecondOrder(self):
        pass

    def classicModelAnalysis(self, ):
        pass

    def cycleRegimeSystem(self):
        pass

    def cycleRegimeSolution(self):
        pass

    def cycleRegimeVisualisation(self):
        pass


if __name__ == "__main__":
    args = {
        'alpha': 0.5,
        'tau_y': 1e3,
        'epsilon': 1e-5,
        'delta': 0.5,
        'lam': 0.5}

    sm = SolowModel(**args)
