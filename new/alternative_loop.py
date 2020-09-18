import time

import numpy as np
import pandas as pd
from line_profiler import LineProfiler
from step_functions import full_general, general, news
from matplotlib import pyplot as plt


def profile(follow=[]):
    def inner(func):
        def profiled_func(*args, **kwargs):
            try:
                profiler = LineProfiler()
                profiler.add_function(func)
                for f in follow:
                    profiler.add_function(f)
                profiler.enable_by_count()
                return func(*args, **kwargs)
            finally:
                profiler.print_stats()

        return profiled_func

    return inner


def alternate_general(t, values: np.ndarray, tech0: float, rho: float,
                      epsilon: float, tau_y: float, tau_s: float, tau_h: float,
                      dep: float, saving0: float, gamma: float, h_h: float,
                      beta1: float, beta2: float, c1: float, c2: float,
                      news: float) -> np.ndarray:
    """ Most general case of the dynamic Solow model including the full set of
    capital demand dynamics and the interplay of demand and supply. The savings
    rate is considered static. The demand system contains the varied feedback
    term

    Parameters
    ----------
    t           :   float
    values      :   np.ndarray
        values at time t, order: y, ks, kd, s, h, gamma_mult,
    tech0, rho, epsilon :   float
        Production parameters
    tau_y, tau_s, tau_h :   float
        Characteristic timescales
    dep, saving0         :   float
        Capital supply parameters
    gamma, h_h          :   float
        Feedback strength
    beta1, beta2        :   float
        Sentiment parameters, beta1>1
    c1, c2, s0          :   float
        Capital demand parameters
    ou_process  :   OrnsteinUhlenbeck

    Returns
    -------
    velocities  :   np.ndarray
    """
    y, ks, kd, s, h, gamma_mult, saving = values
    x = np.zeros(values.shape)

    k = min(ks, kd)

    # Production & Supply
    x[0] = (tech0 * np.exp(rho * k + epsilon * t - y) - 1) / tau_y
    x[1] = (saving0 * np.exp(y - ks) - dep * np.exp(k - ks))

    # Demand System
    gamma_mult_n = 0.5 * (1 + np.tanh(h_h * (ks - kd)))
    feedback = gamma * gamma_mult_n * x[0]
    x[5] = gamma_mult_n - gamma_mult
    x[4] = (-h + np.tanh(feedback + news)) / tau_h
    x[3] = (-s + np.tanh(beta1 * s + beta2 * h)) / tau_s
    x[2] = c1 * x[3] + c2 * s

    return x


@profile(follow=[])
def simulate(t_end, interval, initial_values, params, decay, diffusion, seed):
    # News process
    np.random.seed(seed)
    stoch = np.random.normal(0, np.sqrt(interval), int(t_end / interval))
    stoch2 = np.zeros((int(t_end / interval), 1))
    for i in range(1, int(t_end / interval)):
        stoch2[i] = (1 - interval * decay) * stoch2[i - 1] + diffusion * stoch[
            i]

    values = np.empty((int(t_end / interval), len(start)))
    values[0, :] = initial_values

    for t in range(1, stoch2.shape[0]):
        velocity = alternate_general(t, values[t - 1, :], **params,
                                     news=stoch2[t])
        values[t, :] = values[t - 1, :] + interval * velocity

    x = pd.DataFrame(values, columns=['y', 'ks', 'kd', 's', 'h', 'g', 'news'])
    x.news = stoch2
    return x


@profile(follow=[])
def simulate_gen(t_end, interval, initial_values, params, decay, diffusion,
                 seed):
    # News process
    np.random.seed(seed)
    t_count = int(t_end / interval)
    stoch = np.random.normal(0, np.sqrt(interval), t_count)

    values = np.zeros((t_count, 7), dtype=float)
    values[0, :] = initial_values
    values = full_general(t_end, interval, stoch, values, decay, diffusion,
                          **params)

    return pd.DataFrame(values,
                        columns=['y', 'ks', 'kd', 's', 'h', 'g', 'news'])


@profile(follow=[])
def simulate_part(t_end, interval, initial_values, params, decay, diffusion,
                  seed):
    # News process
    np.random.seed(seed)
    t_count = int(t_end / interval)
    stoch = np.random.normal(0, np.sqrt(interval), t_count)

    xi = news(stoch, stoch.shape[0], decay, diffusion, interval)
    values = np.empty((int(t_end / interval), 7))
    values[0, :] = initial_values

    for t in range(1, xi.shape[0]):
        velocity = general(values[t - 1, :], np.zeros(6), t, **params,
                           news=xi[t])
        for i in range(6):
            values[t, i] = values[t - 1, i] + interval*velocity[i]

    x = pd.DataFrame(values, columns=['y', 'ks', 'kd', 's', 'h', 'g', 'news'])
    x.news = xi
    return x


if __name__ == '__main__':
    params = {
        'tech0': 1, 'rho': 1 / 3, 'epsilon': 1e-5, 'tau_y': 1000,
        'dep': 0.0002,
        "tau_h": 25, "tau_s": 250, "c1": 1, "c2": 3.1e-4, "gamma": 2000,
        "beta1": 1.1, "beta2": 1.0, 'saving0': 0.15, "h_h": 10
    }

    start = np.array([1, 10, 9, 0, 0, 1, 0])
    # Accurate production adjustment
    start[0] = params['epsilon'] + params['rho'] * min(start[1:3])

    t_end = 1e4

    paths = []

    t = time.time()
    path = simulate_gen(t_end=t_end, interval=0.1, initial_values=start,
                        params=params, seed=1, decay=0.2, diffusion=2.0)
    paths.append(path)
    print('######################')
    print("CY-Full Time: {}".format(
            time.strftime("%H:%M:%S", time.gmtime(time.time() - t))))

    t = time.time()
    path = simulate(t_end=t_end, interval=0.1, initial_values=start,
                    params=params, seed=1, decay=0.2, diffusion=2.0)
    paths.append(path)
    print('######################')
    print("PY Time: {}".format(
            time.strftime("%H:%M:%S", time.gmtime(time.time() - t))))


    t = time.time()
    path = simulate_part(t_end=t_end, interval=0.1, initial_values=start,
                          params=params, seed=1, decay=0.2, diffusion=2.0)
    paths.append(path)
    print('######################')
    print("CY Time: {}".format(
            time.strftime("%H:%M:%S", time.gmtime(time.time() - t))))

    fig, ax = plt.subplots(len(paths), 5)
    fig.set_size_inches(8, 4)
    for i, path in enumerate(paths):
        ax[i, 0].plot(path.y)
        ax[i, 1].plot(path.ks, label='ks')
        ax[i, 1].plot(path.kd, label='kd')
        ax[i, 2].plot(path.s)
        ax[i, 3].plot(path.h)
        ax[i, 4].hist(path.news)
    plt.tight_layout()
    plt.show()
    """"""
    print(paths)
