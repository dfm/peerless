from __future__ import division

cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


@cython.boundscheck(False)
def search(double tau,
           np.ndarray[DTYPE_t, ndim=1, mode='c'] t,
           np.ndarray[DTYPE_t, ndim=1, mode='c'] x,
           np.ndarray[DTYPE_t, ndim=1, mode='c'] y,
           np.ndarray[DTYPE_t, ndim=1, mode='c'] ivar):
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
            depth += ivar[i] * y[i]
            depth_ivar += ivar[i]
        if depth_ivar > 0.0:
            depth /= -depth_ivar

        # Save these results.
        out_depth[n] = depth
        out_depth_ivar[n] = depth_ivar

    return out_depth, out_depth_ivar
