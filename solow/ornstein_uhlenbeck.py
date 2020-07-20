import numpy as np


class OrnsteinUhlenbeck(object):
    def __init__(self, decay: float, drift: float, diffusion: float):
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
        self.drift = drift
        self.diffusion = diffusion

        self.history = []
        self.intervals = []

    def euler_maruyama(self, prior: float, delta: float, rand: float = None):
        """ Numerical approximation of the Ornstein-Uhlenbeck process by means
        of the Euler-Maruyama scheme:
        xt+1 = xt + b(xt)*delta_t + diff*sqrt(delta_t)*norm_rand

        Parameters
        ----------
        prior   :   float
            prior realisation of the process
        delta    :   float
            interval delta_t over which the process develops
        rand    :   float
            optionally pass an externally generated random N(0,1) variable

        Returns
        -------
        x   :   float
            next realisation of the Ornstein-Uhlenbeck process
        """
        # Generate a N(0,1) random variable if not provided yet
        rand = rand if rand is not None else np.random.normal(0, 1)

        b_xt = self.decay * (self.drift - prior)
        x_next = prior + delta * b_xt + self.diffusion * np.sqrt(delta) * rand

        return x_next
