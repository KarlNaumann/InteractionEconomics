import numpy as np
cimport numpy as np
from libc.math cimport tanh, exp, sqrt
from cython.view cimport array as cvarray

ctypedef np.float64_t DTYPE_t

def news(double [:] stoch, int size, double decay, double diffusion, double interval):
    cdef np.ndarray[DTYPE_t, ndim=1] news = np.zeros(size)
    cdef int j
    cdef double factor1 = 1 - decay*interval
    news[0] = 0
    for j in range(1, stoch.shape[0]):
        news[j] = factor1*news[j-1] + diffusion*stoch[j];    
    return news

def general(double[:] y, double[:] v, float t, float tech0, float rho, float epsilon, float tau_y,
            float tau_s, float tau_h, float dep, float saving0, float gamma,
            float h_h, float beta1, float beta2, float c1, float c2, float news):

    cdef double k, gamma_mult_n, feedback
    k = min(y[1],y[2])

    # Production & Supply
    v[0] = (tech0 * exp(rho * k + epsilon * t - y[0]) - 1) / tau_y
    v[1] = (saving0 * exp(y[0]-y[1]) - dep * exp(k - y[1]))

    # Demand System
    gamma_mult_n = 0.5 * (1 + tanh(h_h * (y[1]-y[2])))
    feedback = gamma * gamma_mult_n * v[0]
    v[5] = gamma_mult_n - y[5]
    v[4] = (-y[4] + tanh(feedback + news)) / tau_h
    v[3] = (-y[3] + tanh(beta1 * y[3] + beta2 * y[4])) / tau_s
    v[2] = c1 * v[3] + c2 * y[3]

    return np.asarray(v)

def full_general(int t_end, float interval, double [:] stoch, double [:,:] x, float decay, float diffusion, 
                 double tech0, double rho, double epsilon, double tau_y,
                 double tau_s, double tau_h, double dep, double saving0, double gamma,
                 double h_h, double beta1, double beta2, double c1, double c2):
    
    # Set up the main case
    cdef int t = 0
    cdef int i = 0

    # Calculate the news
    cdef double factor = 1 - decay*interval
    for t in range(1, x.shape[0]):
        x[t,6] = factor*x[t-1,6] + diffusion*stoch[t];
    
    # Loop through the main case
    cdef double v[6]
    cdef double k, gamma_mult_n, feedback
    t = 1
    for t in range(1, x.shape[0]):
        v = [0,0,0,0,0,0]
        # Production & Supply
        k = min(x[t-1,1],x[t-1,2])
        v[0] = (tech0 * exp(rho * k + epsilon * t - x[t-1,0]) - 1) / tau_y
        v[1] = (saving0 * exp(x[t-1,0]-x[t-1,1]) - dep * exp(k - x[t-1,1]))

        # Demand System
        gamma_mult_n = 0.5 * (1 + tanh(h_h * (x[t-1,1]-x[t-1,2])))
        feedback = gamma * gamma_mult_n * v[0]
        v[5] = gamma_mult_n - x[t-1,5]
        v[4] = (-x[t-1,4] + tanh(feedback + x[t,6])) / tau_h
        v[3] = (-x[t-1,3] + tanh(beta1 * x[t-1,3] + beta2 * x[t-1,4])) / tau_s
        v[2] = c1 * v[3] + c2 * x[t-1,3]
        
        i = 0
        for i in range(6):
            x[t,i] = x[t-1,i]+interval*v[i]
    
    return np.asarray(x)


