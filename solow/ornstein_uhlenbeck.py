import numpy as np


class OrnsteinUhlenbeck(object):
    def __init__(self, decay: float, drift: float, diffusion: float,
                 t0: float = 0):
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
        self.drift = drift
        self.diffusion = diffusion

        self.history = [self.drift]
        self.points = [t0]

    def euler_maruyama(self, t: float, rand: float = None):
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
        # Generate a N(0,1) random variable if not provided yet
        rand = rand if rand is not None else np.random.normal(0, 1)

        # Determine the interval
        delta = t - self.points[-1]
        self.points.append(t)

        # Next realisation as function of prior realisation
        prior = self.history[-1]
        b_xt = self.decay * (self.drift - prior)
        x_next = prior + delta * b_xt + self.diffusion * np.sqrt(delta) * rand
        self.history.append(x_next)

        return x_next
