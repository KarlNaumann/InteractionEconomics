#!python
#cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport tanh, exp, sqrt

def long_general(float interval, int count, double [:] stoch, double [:,:] x, float decay, float diffusion,
                 double tech0, double rho, double epsilon, double tau_y,
                 double tau_s, double tau_h, double dep, double saving0, double gamma,
                 double h_h, double beta1, double beta2, double c1, double c2):

    # Set up the main case
    cdef int i
    cdef int j
    cdef double v[7]
    cdef double p[7]
    cdef double k
    cdef double g_mult

    # Initial values
    for j in range(7):
        p[j] = x[0,j]

    # Loop through the main case
    for i in range(1, x.shape[0]*count):
        v = [0,0,0,0,0,0,0]

        # Production & Supply
        k = min(p[1],p[2])
        v[0] = (tech0 * exp(rho * k + epsilon * i*interval - p[0]) - 1) / tau_y
        v[1] = (saving0 * exp(p[0]-p[1]) - dep * exp(k-p[1]))

        # Demand System
        g_mult = 0.5 * (1 + tanh(h_h * (p[1]-p[2])))
        v[5] = g_mult - p[5]
        v[4] = (-p[4] + tanh(gamma * g_mult * v[0] + p[6])) / tau_h
        v[3] = (-p[3] + tanh(beta1 * p[3] + beta2 * p[4])) / tau_s
        v[2] = c1 * v[3] + c2 * p[3]
        v[6] = -1*decay*p[6] + (diffusion*stoch[i]/sqrt(interval))

        for j in range(7):
            p[j] = p[j]+interval*v[j]

        # Save in the times where t is int
        if i % count == 0:
            for j in range(7):
                x[i//count,j] = p[j]

    return np.asarray(x)

def demand_case(float interval, int count, double [:] stoch, double [:,:] x,
                float decay, float diffusion, double tech0, double rho,
                double epsilon, double tau_y, double tau_s, double tau_h,
                double gamma, double beta1, double beta2, double c1,
                double c2, double s0):

    # Set up the main case
    cdef int i
    cdef int j
    cdef double v[6]
    cdef double p[6]

    # Initial values (Order: y, z, kd, s, h, xi)
    for j in range(6):
        p[j] = x[0,j]

    # Loop through the main case
    for i in range(1, x.shape[0]*count):
        v = [0,0,0,0,0,0]
        # Production
        v[0] = (tech0 * exp(p[1]) - 1) / tau_y
        # Information and Sentiment
        v[4] = (-p[4] + tanh(gamma * v[0] + p[5])) / tau_h
        v[3] = (-p[3] + tanh(beta1 * p[3] + beta2 * p[4])) / tau_s
        # Demand and Z
        v[2] = c1 * v[3] + c2 * (p[3] - s0)
        v[1] = v[2] - v[0] + epsilon
        # News
        v[5] = -1*decay*p[5] + (diffusion*stoch[i]/sqrt(interval))
        # Update
        for j in range(6):
            p[j] = p[j]+interval*v[j]
        # Save in the times where t is integer valued
        if i % count == 0:
            for j in range(6):
                x[i//count,j] = p[j]

    return np.asarray(x)


