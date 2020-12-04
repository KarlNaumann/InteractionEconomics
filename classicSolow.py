import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib import rc
from scipy.integrate import solve_ivp
from scipy.integrate import odeint

def boundary_layer_approximation(t_end: float, b: float, eps: float, rho: float, tau_y: float, lam: float, dep: float):
    """ Calculate the path of production for the classic Solow case based on
    the approximate solution from the boundary layer technique

    Parameters
    ----------
    t_end   :   float
        duration of the simulation
    b   :   float
        constant of integration
    eps :   float
        technology growth rate
    rho :   float
        capital share in cobb-douglas production
    tau_y   :   float
        characteristic timescale of production
    lam :   float
        household saving rate
    dep :   float
        depreciation rate

    Returns
    -------
    solution    :    np.ndarray
        solution path of production

    """
    rhoi = 1 - rho
    constant = (lam / dep) ** (rho / rhoi)
    t = np.arange(int(t_end))
    temp = (b * np.exp(-rhoi * t / tau_y) + 1) ** (1 / rhoi)
    temp += np.exp(eps * t / rhoi)
    return constant * (temp - 1)


def classic_solow_growth_path(t_end: float, start: list, eps: float, rho: float, tau_y: float, lam: float, dep: float):
    """ Function to integrate the path of capital and production in the classic
    solow limiting case

    Parameters
    ----------
    t_end   :   float
        total time of the simulation
    start   :   list
        initial values y0, k0
    eps :   float
        technology growth rate
    rho :   float
        capital share in cobb-douglas production
    tau_y   :   float
        characteristic timescale of production
    lam :   float
        household saving rate
    dep :   float
        depreciation rate

    Returns
    -------
    path    :   pd.DataFrame
        path of capital and production
    """

    path = pd.DataFrame(index=np.arange(int(t_end)), columns=['y', 'k'], dtype=float)
    path.loc[0, :] = start
    for t in path.index[1:]:
        y, k = path.loc[t - 1, :].values
        v_y = np.exp((rho * k) + (eps * t) - y) - 1
        v_k = lam * np.exp(y - k) - dep
        path.loc[t, 'y'] = path.loc[t-1,'y'] + v_y / tau_y
        path.loc[t, 'k'] = path.loc[t-1,'k'] + v_k
    return path


def second_order_differential(t_end: float, start: list, y0: float, eps: float, rho: float, tau_y: float, lam: float, dep: float):
    """ Function to integrate the path of capital and of capital and production
    based on the second-order differential equation in the classic Solow case

    Parameters
    ----------
    t_end   :   float
        total time of the simulation
    start   :   list
        initial values k0, y0
    eps :   float
        technology growth rate
    rho :   float
        capital share in cobb-douglas production
    tau_y   :   float
        characteristic timescale of production
    lam :   float
        household saving rate
    dep :   float
        depreciation rate

    Returns
    -------
    path    :   pd.DataFrame
        path of capital and production
    """

    path_k = pd.DataFrame(index=np.arange(int(t_end)), columns=['k', 'dk'])
    path_k.loc[0, :] = start
    for t in path_k.index[1:]:
        k, dk_dt = path_k.loc[t-1,:]
        d2k_dt2 = lam * (k ** rho) * np.exp(eps * t)
        d2k_dt2 -= dep * k
        d2k_dt2 -= (1 + tau_y * dep) * dk_dt
        d2k_dt2 = d2k_dt2 #/ tau_y

        path_k.loc[t,'k'] = k + dk_dt
        path_k.loc[t,'dk'] = dk_dt + d2k_dt2

    y = exact_production(path_k.loc[:,'k'].values, y0, eps, rho, tau_y)
    data = np.hstack([path_k.loc[:,'k'].values[:, np.newaxis], y[:, np.newaxis]])
    return pd.DataFrame(data, columns=['K', 'Y']).astype(float)



def exact_step(t: float, x: list, eps: float, rho: float, tau_y: float, lam: float, dep: float):
    """ Velocities for the second order differential equation that summarises
    the path of capital in the classic solow limiting case

    Parameters
    ----------
    t   :   float
        time of the current step
    x   :   list
        variables k and dk_dt at time t
    eps :   float
        technology growth rate
    rho :   float
        capital share in cobb-douglas production
    tau_y   :   float
        characteristic timescale of production
    lam :   float
        household saving rate
    dep :   float
        depreciation rate

    Returns
    -------
    velocity    :   list
        dk_dt and d2k_dt2 (first and second order derivative at t
    """
    k, dk_dt = x

    d2k_dt2 = lam * (k ** rho) * np.exp(eps * t)
    d2k_dt2 -= dep * k
    d2k_dt2 -= (1 + tau_y * dep) * dk_dt
    d2k_dt2 = d2k_dt2 / tau_y

    return [dk_dt, d2k_dt2]


def exact_production(k: np.ndarray, y0: float, eps: float, rho: float, tau_y: float):
    """ Calculating the path of production given the path of capital determined
    byt he second order differential equation

    Parameters
    ----------
    k   :   np.ndarray
        path of capital
    eps :   float
        technology growth rate
    rho :   float
        capital share in production
    tau_y   :   float
        characteristic timescale of production

    Returns
    -------
    y   :   np.ndarray
        path of production
    """

    y = np.empty_like(k)
    y[0] = y0
    for t in range(1, y.shape[0]):
        v_y = np.exp(eps * t) * k[t] ** rho - y[t - 1]
        y[t] = y[t - 1] + (v_y / tau_y)

    return y


def exact_solution(t_end: float, start: list, y0: float, eps: float, rho: float, tau_y: float, lam: float, dep: float):
    """ Function to integrate the path of capital and of capital and production
    based on the second-order differential equation in the classic Solow case

    Parameters
    ----------
    t_end   :   float
        total time of the simulation
    start   :   list
        initial values k0, y0
    eps :   float
        technology growth rate
    rho :   float
        capital share in cobb-douglas production
    tau_y   :   float
        characteristic timescale of production
    lam :   float
        household saving rate
    dep :   float
        depreciation rate

    Returns
    -------
    path    :   pd.DataFrame
        path of capital and production
    """

    def derivative(x, t, eps, rho, tau, lam, dep):
        k, dk = x
        dk2 = lam * (k**rho) * np.exp(eps*t) - (dep*k) - (1+tau*dep) * dk
        return np.array([dk, dk2/tau])

    t = np.arange(int(t_end))
    args = (eps, rho, tau_y, lam, dep)
    k,_ = odeint(derivative, start, t, args=args).T
    y = exact_production(k, y0, eps, rho, tau_y)
    data = np.hstack([k[:,np.newaxis], y[:,np.newaxis]])
    return pd.DataFrame(data, columns=['k', 'y'], dtype=float)

    """
    path = solve_ivp(exact_step, t_span=(0, t_end), y0=start,
                     t_eval=np.arange(int(t_end)),
                     args=(rho, eps, dep, lam, tau_y),
                     atol=1e-5, rtol=1e-5, max_step=100, first_step=1)

    y = exact_production(path.y.T[:,0], y0, eps, rho, tau_y)
    data = np.hstack([path.y.T[:,0, np.newaxis], y[:, np.newaxis]])
    return pd.DataFrame(data, columns=['k', 'y'], dtype=float)
    """

def plot_settings():
    # Plotting preamble
    sns.set()
    plt.rcParams['text.latex.preamble']\
        = r'\usepackage[bitstream-charter, greekfamily=default]{mathdesign}'
    rc('text', usetex=True)
    rc('font', **{'family': 'serif'})
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams.update({'figure.figsize': (8, 6),
                         'axes.titlesize': 18,
                         'legend.fontsize': 18,
                         'axes.labelsize': 20})


def time_series_plot(df: pd.DataFrame, save: str = ''):
    fig, ax = plt.subplots(1, 1)
    for col in df.columns:
        ax.plot(df.loc[:, col], label=col)
    ax.legend()
    fig.tight_layout()
    if save != '':
        if '.png' not in save:
            save += '.png'
        plt.savefig(save, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':

    #plot_settings()

    # Parameterisation
    p = dict(rho=0.5, eps=1/100000, tau_y=1000, lam=0.5, dep=0.5)
    const = 1e-1
    t_end = 4e4

    # Boundary layer approximation
    bla = boundary_layer_approximation(t_end, const, **p)

    # Starting values are in real terms and match the integral
    ln_y0 = np.log(bla[0])
    ln_k0 = ln_y0 / p['rho']
    print("Starting values:\n\ty = {:.3f}\n\tk = {:.3f}".format(ln_y0, ln_k0))

    solow = np.exp(classic_solow_growth_path(t_end, [ln_y0, ln_k0], **p))
    exact = exact_solution(t_end, [np.exp(ln_k0), 0], bla[0], **p)

    data_y = np.hstack(
            [bla[:, np.newaxis], solow.loc[:, 'y'].values[:,np.newaxis],
             exact.y.values[:, np.newaxis]])
    comparison_y = pd.DataFrame(data_y, columns=['BL', 'Solow', 'Exact'])
    print(comparison_y.head())

    data_k = np.hstack([solow.loc[:, 'k'].values[:,np.newaxis],
                       exact.k.values[:, np.newaxis]])
    comparison_k = pd.DataFrame(data_k, columns=['Solow', 'Exact'])
    print(comparison_k.head())
    time_series_plot(comparison_y)
