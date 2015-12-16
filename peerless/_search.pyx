from __future__ import division

cimport cython
from libc.math cimport isnan

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


@cython.boundscheck(False)
def search(double tau,
           np.ndarray[DTYPE_t, ndim=1, mode='c'] t,
           np.ndarray[DTYPE_t, ndim=1, mode='c'] x,
           np.ndarray[DTYPE_t, ndim=1, mode='c'] y,
           np.ndarray[DTYPE_t, ndim=1, mode='c'] ivar,
           double apodize=1.0):
    cdef int i0 = 0
    cdef int n, i
    cdef int nt = t.shape[0]
    cdef int nx = x.shape[0]
    cdef double tmn, tmx
    cdef double depth, depth_ivar
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] out_depth = np.empty(nt, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] out_depth_ivar = np.empty(nt, dtype=DTYPE)

    for n in range(nt):
        tmn = t[n]
        tmx = tmn + tau
        depth = 0.0
        depth_ivar = 0.0

        # Find the starting coordinate.
        for i in range(i0, nx):
            if x[i] >= tmn:
                i0 = i
                break

        # Compute the model online.
        for i in range(i0, nx):
            if x[i] >= tmx:
                break

            # Any missing points.
            if isnan(y[i]):
                depth = 0.0
                depth_ivar = 0.0
                break

            # Compute the depth online.
            depth += ivar[i] * y[i]
            depth_ivar += ivar[i]

        if depth_ivar > 0.0:
            depth /= -depth_ivar

        # Save these results.
        out_depth[n] = depth
        out_depth_ivar[n] = depth_ivar

    if apodize > 0.0:
        out_depth /= 1.0 + np.exp(-(t - (t[0] + apodize * tau)) / tau)
        out_depth /= 1.0 + np.exp((t - (t[-1] - apodize * tau)) / tau)

    return out_depth, out_depth_ivar
