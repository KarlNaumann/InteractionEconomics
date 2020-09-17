import numpy as np


class OrnsteinUhlenbeck(object):
    def __init__(self, decay: float, diffusion: float, t0: float = 0):
        """ Class implementing the Ornstein-Uhlenbeck process

            dxt = decay * (drift - xt) * dt + diffusion * dWt

        by various means of approximation

        Parameters
        ----------
        decay   :   float
            decay parameter (also known as theta)
        drift   :   float
            asymptotic mean of the process
        diffusion   :   float
            diffusion coefficient
        t0  :   float
            starting time for the process

        Public Attributes
        -----------------
        history :   list
            realisations of this instance of the Ornstein-Uhlenbeck process
        intervals   :   list
            intervals for the Ornstein-Uhlenbeck process

        Public Methods
        --------------
        euler_maruyama()
            approximate the next realisation by means of the Euler Maruyama
            approach
        """

        self.decay = decay
        self.diffusion = diffusion

        self.history = [0]
        self.points = [t0]

    def euler_maruyama(self, t: float):
        """ Numerical approximation of the Ornstein-Uhlenbeck process by means
        of the Euler-Maruyama scheme:
        xt+1 = xt + b(xt)*delta_t + diff*sqrt(delta_t)*norm_rand

        Parameters
        ----------
        t    :   float
            time t where the next evaluation occurs
        rand    :   float
            optionally pass an externally generated random N(0,1) variable

        Returns
        -------
        x   :   float
            next realisation of the Ornstein-Uhlenbeck process
        """
        # ODE algo may be adaptive and go backwards in between -> find the
        # closest timestep that is smaller
        i = 1
        while self.points[-i] >= t:
            i += 1
            if i > len(self.points):
                i = len(self.points)
                break

        # Determine the interval
        delta = t - self.points[-i]
        self.points.append(t)

        const = (1 - delta * self.decay) * self.history[-1]
        xi_next = const + self.diffusion * np.random.normal(0, np.sqrt(delta))
        self.history.append(xi_next)
        return xi_next
