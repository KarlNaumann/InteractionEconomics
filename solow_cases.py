""" File containing all the different step functions that are tested """

import numpy as np

from ornstein_uhlenbeck import OrnsteinUhlenbeck


def general(t, values: np.ndarray, tech0: float, rho: float, epsilon: float,
            tau_y: float, tau_s: float, tau_h: float, dep: float,
            saving0: float,
            gamma: float, h_h: float, beta1: float, beta2: float, c1: float,
            c2: float, s0: float,
            ou_process: OrnsteinUhlenbeck = None) -> np.ndarray:
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

    if ou_process is not None:
        news = ou_process.euler_maruyama(t)
    else:
        news = 0

    k = kd - (kd - ks) * 0.5 * (1 + np.tanh(10 * (kd - ks)))

    # Production & Supply
    v_y = (np.log(tech0) * np.exp(rho * k + epsilon * t - y) - 1) / tau_y
    v_saving = 0
    v_ks = (saving0 * np.exp(y - ks) - dep * np.exp(k - ks))

    # Demand System
    gamma_mult_n = 0.5 * (1 + np.tanh(h_h * (ks - kd)))
    feedback = gamma * gamma_mult_n * v_y
    v_g_s = gamma_mult_n - gamma_mult
    v_h = (-h + np.tanh(feedback + news)) / tau_h
    v_s = (-s + np.tanh(beta1 * s + beta2 * h)) / tau_s
    v_kd = c1 * v_s + c2 * (s - s0)

    return np.array([v_y, v_ks, v_kd, v_s, v_h, v_g_s, v_saving])


def unbounded_information(t, values: np.ndarray, tech0: float, rho: float,
                          epsilon: float, tau_y: float, tau_s: float,
                          tau_h: float, dep: float, saving0: float,
                          gamma: float, h_h: float, beta1: float, beta2: float,
                          c1: float, c2: float, s0: float,
                          ou_process: OrnsteinUhlenbeck = None) -> np.ndarray:
    """ Most general case of teh dynamic Solow model including the full set of
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

    if ou_process is not None:
        news = ou_process.euler_maruyama(t)
    else:
        news = 0

    k = kd - (kd - ks) * 0.5 * (1 + np.tanh(10 * (kd - ks)))

    # Production & Supply
    v_y = (np.log(tech0) * np.exp(rho * k + epsilon * t - y) - 1) / tau_y
    v_saving = 0
    v_ks = (saving0 * np.exp(y - ks) - dep * np.exp(k - ks))

    # Demand System
    gamma_mult_n = 0.5 * (1 + np.tanh(h_h * (ks - kd)))
    feedback = gamma * gamma_mult_n * v_y
    v_g_s = gamma_mult_n - gamma_mult
    v_h = (-h + feedback + news) / tau_h
    v_s = (-s + np.tanh(beta1 * s + beta2 * h)) / tau_s
    v_kd = c1 * v_s + c2 * (s - s0)

    return np.array([v_y, v_ks, v_kd, v_s, v_h, v_g_s, v_saving])


def general_no_patch(t, values: np.ndarray, tech0: float, rho: float,
                     epsilon: float, tau_y: float, tau_s: float, tau_h: float,
                     dep: float, saving0: float, gamma: float, h_h: float,
                     beta1: float, beta2: float, c1: float, c2: float,
                     s0: float,
                     ou_process: OrnsteinUhlenbeck = None) -> np.ndarray:
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

    if ou_process is not None:
        news = ou_process.euler_maruyama(t)
    else:
        news = 0

    k = kd - (kd - ks) * 0.5 * (1 + np.tanh(10 * (kd - ks)))

    # Production & Supply
    v_y = (np.log(tech0) * np.exp(rho * k + epsilon * t - y) - 1) / tau_y
    v_saving = 0
    v_ks = (saving0 * np.exp(y - ks)) - (dep * np.exp(k - ks))

    # Demand System
    gamma_mult_n = 1.0
    feedback = gamma * gamma_mult_n * v_y
    v_g_s = 0
    v_h = (-h + np.tanh(feedback + news)) / tau_h
    v_s = (-s + np.tanh(beta1 * s + beta2 * h)) / tau_s
    v_kd = c1 * v_s + c2 * (s - s0)

    return np.array([v_y, v_ks, v_kd, v_s, v_h, v_g_s, v_saving])


# FEEDBACK LOCATIONS


def direct_feedback(t, values: np.ndarray, tech0: float, rho: float,
                    epsilon: float, tau_y: float, tau_s: float, tau_h: float,
                    dep: float, saving0: float, gamma: float, h_h: float,
                    beta1: float, beta2: float, c1: float, c2: float, s0: float,
                    ou_process: OrnsteinUhlenbeck = None) -> np.ndarray:
    """ Most general case of teh dynamic Solow model including the full set of
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

    if ou_process is not None:
        news = ou_process.euler_maruyama(t)
    else:
        news = 0

    k = kd - (kd - ks) * 0.5 * (1 + np.tanh(10 * (kd - ks)))

    # Production & Supply
    v_y = (np.log(tech0) * np.exp(rho * k + epsilon * t - y) - 1) / tau_y
    v_saving = 0
    v_ks = (saving0 * np.exp(y - ks) - dep * np.exp(k - ks))

    # Demand System
    gamma_mult_n = 0.5 * (1 + np.tanh(h_h * (ks - kd)))
    feedback = gamma * gamma_mult_n * v_y
    v_g_s = gamma_mult_n - gamma_mult
    v_h = (-h + np.tanh(news)) / tau_h
    v_s = (-s + np.tanh(beta1 * s + beta2 * h)) / tau_s
    v_kd = c1 * v_s + c2 * (s - s0) + feedback

    return np.array([v_y, v_ks, v_kd, v_s, v_h, v_g_s, v_saving])


def sentiment_feedback(t, values: np.ndarray, tech0: float, rho: float,
                       epsilon: float, tau_y: float, tau_s: float, tau_h: float,
                       dep: float, saving0: float, gamma: float, h_h: float,
                       beta1: float, beta2: float, c1: float, c2: float,
                       s0: float,
                       ou_process: OrnsteinUhlenbeck = None) -> np.ndarray:
    """ Most general case of teh dynamic Solow model including the full set of
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

    if ou_process is not None:
        news = ou_process.euler_maruyama(t)
    else:
        news = 0

    k = kd - (kd - ks) * 0.5 * (1 + np.tanh(10 * (kd - ks)))

    # Production & Supply
    v_y = (np.log(tech0) * np.exp(rho * k + epsilon * t - y) - 1) / tau_y
    v_saving = 0
    v_ks = (saving0 * np.exp(y - ks) - dep * np.exp(k - ks))

    # Demand System
    gamma_mult_n = 0.5 * (1 + np.tanh(h_h * (ks - kd)))
    feedback = gamma * gamma_mult_n * v_y
    v_g_s = gamma_mult_n - gamma_mult
    v_h = (-h + np.tanh(news)) / tau_h
    v_s = (-s + np.tanh(beta1 * s + beta2 * h + feedback)) / tau_s
    v_kd = c1 * v_s + c2 * (s - s0)

    return np.array([v_y, v_ks, v_kd, v_s, v_h, v_g_s, v_saving])


# LIMITING CASES FOR SUPPLY & DEMAND


def limit_kd(t, values: np.ndarray, tech0: float, rho: float,
             epsilon: float, tau_y: float, tau_s: float, tau_h: float,
             dep: float, saving0: float, gamma: float, h_h: float,
             beta1: float, beta2: float, c1: float, c2: float, s0: float,
             ou_process: OrnsteinUhlenbeck = None) -> np.ndarray:
    """ Most general case of teh dynamic Solow model including the full set of
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

    if ou_process is not None:
        news = ou_process.euler_maruyama(t)
    else:
        news = 0

    k = kd

    # Production & Supply
    v_y = (np.log(tech0) * np.exp(rho * k + epsilon * t - y) - 1) / tau_y
    v_saving = 0
    v_ks = (saving0 * np.exp(y - ks) - dep * np.exp(k - ks))

    # Demand System
    gamma_mult_n = 0.5 * (1 + np.tanh(h_h * (ks - kd)))
    feedback = gamma * gamma_mult_n * v_y
    v_g_s = gamma_mult_n - gamma_mult
    v_h = (-h + np.tanh(news)) / tau_h
    v_s = (-s + np.tanh(beta1 * s + beta2 * h)) / tau_s
    v_kd = c1 * v_s + c2 * (s - s0) + feedback

    return np.array([v_y, v_ks, v_kd, v_s, v_h, v_g_s, v_saving])


def limit_ks(t, values: np.ndarray, tech0: float, rho: float,
             epsilon: float, tau_y: float, tau_s: float, tau_h: float,
             dep: float, saving0: float, gamma: float, h_h: float,
             beta1: float, beta2: float, c1: float, c2: float, s0: float,
             ou_process: OrnsteinUhlenbeck = None) -> np.ndarray:
    """ Most general case of teh dynamic Solow model including the full set of
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

    if ou_process is not None:
        news = ou_process.euler_maruyama(t)
    else:
        news = 0

    k = ks

    # Production & Supply
    v_y = (np.log(tech0) * np.exp(rho * k + epsilon * t - y) - 1) / tau_y
    v_saving = 0
    v_ks = (saving0 * np.exp(y - ks) - dep * np.exp(k - ks))

    # Demand System
    gamma_mult_n = 0.5 * (1 + np.tanh(h_h * (ks - kd)))
    feedback = gamma * gamma_mult_n * v_y
    v_g_s = gamma_mult_n - gamma_mult
    v_h = (-h + np.tanh(news)) / tau_h
    v_s = (-s + np.tanh(beta1 * s + beta2 * h)) / tau_s
    v_kd = c1 * v_s + c2 * (s - s0) + feedback

    return np.array([v_y, v_ks, v_kd, v_s, v_h, v_g_s, v_saving])
