import numpy as np
from scipy.optimize import minimize

from ornstein_uhlenbeck import OrnsteinUhlenbeck


class DynamicDemandSystem(object):
    def __init__(self, tau_h: float = 25, tau_s: float = 250,
                 tau_y: float = 2000, epsilon: float = 5e-5, c1: float = 1,
                 c2: float = 15e-5, s0: float = 0, tech0: float = 1,
                 beta1: float = 1.1, beta2: float = 1.0, gamma: float = 2000,
                 theta: float = 0.2, sigma: float = 2.5,
                 ou: OrnsteinUhlenbeck = None):
        """ Class instance of the generic dynamic capital demand system in the
        limiting case of Ks>Kd for all t.

        Parameters
        ----------
        tau_h, tau_s, tau_y   :   float (defaults: 25, 250, 2000)
            characteristic timescales of information, sentiment, and production
        epsilon :   float (default: 5e-5)
            technology growth rate
        c1, c2, s0  :   float (defaults: 1, 15e-5)
            coefficients of capital demand (change in sentiment, sentiment,
            long-term sentiment)
        tech0   :   float
            Initial level of technology
        beta1, beta2    :   float (defaults: 1.1, 1.0)
            coefficients of sentiment (interactions and information)
        gamma   :   float (default: 2000)
            strength of production feedback
        phi :   float (default: 2000)
            strength of the excess capital feedback
        theta, sigma    :   float (defaults: 0.2, 2.5)
            parametrisation of the Ornstein-Uhlenbeck white noise process
        """
        # Arguments on a per function basis
        self.z_args = {
            "tau": tau_y, "eps": epsilon, "c1": c1, "c2": c2, "s0": s0,
            "tech0": tech0
        }
        self.s_args = {"tau": tau_s, "b1": beta1, "b2": beta2}
        self.h_args = {"tau": tau_h, "gamma": gamma}
        self.ou_args = {"decay": theta, "diffusion": sigma, "drift": 0, "t0": 1}

        # Initialise some additional useful params
        self.epsilon = epsilon
        self.ou = ou if ou is not None else OrnsteinUhlenbeck(**self.ou_args)

    def velocity(self, t, x, use_k=True, ou=None) -> np.ndarray:
        """ Calculate the velocity of the demand system in (s,h,z)-space.
        Function is static so that it can be used in the solve_ivp optimiser

        Parameters
        ----------
        x   :   list
            values of (s,h,z) in that order
        times, demand, s_arg, h_arg   :   dict
            dictionaries of the relevant parametrisation
        use_k   :   bool
            whether to calculate the derivative in k as well

        ou  :   OrnsteinUhlenbeck
            Ornstein-Uhlenbeck white noise process instance

        Returns
        -------
        v_x :   list
            velocities of s,h,z at that point in time
        """

        s, h, z = x[0], x[1], x[2]

        # Change in Sentiment
        v_s = -s + np.tanh(self.s_args['b1'] * s + self.s_args['b2'] * h)
        v_s = v_s / self.s_args['tau']

        # Change in Information
        delta_prod = (self.z_args['tech0'] * np.exp(z) - 1) / self.z_args['tau']
        force = self.h_args['gamma'] * delta_prod

        if ou is not None:
            force += ou.euler_maruyama(t)

        v_h = (-h + np.tanh(force)) / self.h_args['tau']

        v_z = self.z_args['c1'] * v_s + self.z_args['c2'] * s - delta_prod \
              + self.epsilon

        if use_k:
            v_k = self.z_args['c1'] * v_s + self.z_args['c2'] * s
            return np.array([v_s, v_h, v_z, v_k])
        else:
            return np.array([v_s, v_h, v_z])

    def critical_points(self):
        """ Determine the critical points in the system, where v_s, v_h and v_z
        are equal to 0. Do this by substituting in for s, solving s, and then
        solving the remaining points.

        Returns
        -------
        coordinates :   list
            list of tuples for critical coordinates in (s,h,z)-space
        """

        # minimisation function to find where this holds i.e. = 0
        def f(s, b1, b2, gamma, c2, s0, epsilon):
            inner = gamma * (c2 * (s - s0) + epsilon)
            return np.sqrt((np.tanh(b1 * s + b2 * np.tanh(inner)) - s) ** 2)

        sols = []
        filtered = []
        min_options = {'eps': 1e-10}
        args = (
            self.s_args["b1"], self.s_args["b1"], self.h_args["gamma"],
            self.z_args["c2"], self.z_args["s0"], self.epsilon)

        # Use minimiser to determine where the function crosses 0 (s_dot=0)
        for x in np.linspace(-1, 1, 11):
            candidate = minimize(f, x0=x, bounds=[(-1, 1)], args=args,
                                 method='L-BFGS-B', options=min_options)
            if candidate.success:
                # Check if this critical point is already in the solution list
                if all([np.abs(sol-candidate.x[0])>1e7 for sol in sols]):
                    filtered.append(candidate.x[0])
                sols.append(candidate.x[0])

        print(filtered)
        print(sols)

        # Eliminate duplicated solutions (artefact of multiple x0 in minimise)
        filtered = []
        for i, val in enumerate(sols):
            found = False
            for j in range(i + 1, len(sols)):
                if np.sum(np.abs(sols[j] - val)) < 1e-7:
                    found = True
                    break
            if not found:
                filtered.append(val)

        # Determine h and z for each critical point in s
        coordinates = []
        for i, s in enumerate(filtered):
            inner = (self.z_args["c2"] * (s - self.z_args['s0']) + self.epsilon)
            h = np.tanh(self.h_args['gamma'] * inner)
            z = np.log((self.z_args['tau'] * inner + 1) / self.z_args['tech0'])
            coordinates.append((s, h, z))

        return coordinates

    def overview(self, t0: 1, t_end=1e5):
        """Generate three overview plots: (1) the phase diagram, (2) time-series
        plot of sentiment, (3) 3D plot of (s,h,z) over time

        Parameters
        ----------
        t0, t_end
        """

    def simulate(self, init_val_list: list, t0: float = 1, t_end: float = 2e5):
        """ Simulate the system given the initialised parameters

        Parameters
        ----------
        init_val_list   :   list
            initial values in the (s,h,z,k) space
        t0, t_end  :   float
            start and end times of the simulation

        Returns
        -------
        paths   :   dict
            dataframe of the
        """
