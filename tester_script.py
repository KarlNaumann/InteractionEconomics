# Test file

import numpy as np
from matplotlib import pyplot as plt

from classicSolow import ClassicSolow
from solowModel import SolowModel

# Initialise parameters
params = dict(tech0=1, rho=1 / 3, epsilon=1e-5, tau_y=1000, dep=0.0002,
              tau_h=25, tau_s=250, c1=1, c2=2.5e-4, gamma=1000, beta1=1.1,
              beta2=1.0, saving0=0.15, h_h=10)

xi_args = dict(decay=0.2, diffusion=2.0)
start = np.array([1, 10, 9, 0, 0, 1, params['saving0']])
start[0] = params['epsilon'] + params['rho'] * min(start[1:3])

sm = SolowModel(params=params, xi_args=xi_args)

general = False
supply_limit = True

if general:
    # General case
    df = sm.simulate(start, t_end=1e7, seed=0)
    sm.visualise()

if supply_limit:
    cs = ClassicSolow(params)
    cs.simulate([20, 3], 1e5)
    cs.visualise(save='figures/fig_limitks')
