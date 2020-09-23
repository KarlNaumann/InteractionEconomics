# Test file

import numpy as np
import time
import pickle

from matplotlib import pyplot as plt

from solowModel_cython import SolowModel

params = dict(tech0=1, rho=0.33333, epsilon=1e-5, tau_y=2000, dep=0.0002,
                  tau_h=25, tau_s=250, c1=1, c2=2e-4, gamma=1000, beta1=1.1,
                  beta2=1.0, saving0=0.15, h_h=10)

xi_args = dict(decay=0.2, diffusion=2.0)
start = np.array([1, 10, 9, 0, 0, 1, params['saving0']])
start[0] = params['epsilon'] + params['rho'] * min(start[1:3])

sm = SolowModel(params=params, xi_args=xi_args)
df = sm.simulate(start, t_end=1e6, seed=0)

print(df.head())
fig, ax = plt.subplots(2,2)
ax[0,0].plot(df.y)
ax[0,1].plot(df.ks)
ax[0,1].plot(df.kd)
ax[1,0].plot(df.s)
ax[1,1].plot(df.h)
plt.title('Regular')
plt.tight_layout()
plt.show(block=False)


t = time.time()
df = sm.simulate(start, t_end=1e7, seed=0)
df2 = sm.asymptotics()
file = open('speed_test_export.df', 'wb')
pickle.dump(df, file)
file.close()
print("Time: {}".format(time.strftime("%H:%M:%S", time.gmtime(time.time() - t))))
print(df.head())

fig, ax = plt.subplots(2,2)
ax[0,0].plot(df.y)
ax[0,1].plot(df.ks)
ax[0,1].plot(df.kd)
ax[1,0].plot(df.s)
ax[1,1].plot(df.h)
plt.title('Long')
plt.tight_layout()
plt.show()