# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["fit_gp", "fit_box", "fit_step", "fit_transit"]

from itertools import product

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import minimize, OptimizeResult

import transit
import george
from george.kernels import Matern32Kernel


def fit_gp(x, y, yerr):
    # Remove missing data.
    m = np.isfinite(y) & np.isfinite(y)
    ym = y[m] - 1.0

    # Initialize the GP.
    gp = init_gp(x[m], ym, yerr)

    # Define the model.
    def nll(p):
        gp.kernel[:] = p
        return -gp.lnlikelihood(ym, quiet=True)

    def grad_nll(p):
        gp.kernel[:] = p
        return -gp.grad_lnlikelihood(ym, quiet=True)

    # Optimize.
    p0 = gp.kernel.vector
    init = nll(p0)
    result = minimize(nll, p0, jac=grad_nll, method="L-BFGS-B",
                      bounds=[(None, 0.0), (-10.0, None)])
    gp.kernel[:] = result.x

    # Save the results.
    result.initial_lnlike = -init
    result.gp = gp
    result.lnlike = -result.fun

    # Compute prediction.
    bkg = result.gp.predict(ym, x[m], mean_only=True)
    result.prediction = 1.0 + bkg

    return result


def fit_box(x, y, yerr, t0s, taus, gp=None):
    if gp is None:
        gp = init_gp(x, y, yerr)

    # Remove missing data.
    m = np.isfinite(y) & np.isfinite(y)
    xm, ym = x[m], y[m]

    # Define the model.
    def model((t0, tau), x):
        m = np.abs(x-t0) < 0.5*tau
        if not m.sum():
            return None
        mod = np.zeros_like(x)
        mod[m] = -1.0
        return mod

    return fit_linear(xm, ym, gp, model, product(t0s, taus))


def fit_step(x, y, yerr, t0s, gp=None):
    if gp is None:
        gp = init_gp(x, y, yerr)

    # Remove missing data.
    m = np.isfinite(y) & np.isfinite(y)
    xm, ym = x[m], y[m]

    # Define the model.
    def model(t0, x):
        m = x < t0
        if not m.sum():
            return None
        mod = np.zeros_like(x)
        mod[m] = 1.0
        return mod

    return fit_linear(xm, ym, gp, model, t0s)


def fit_linear(x, y, gp, model, parameters):
    # Iterate over times and durations.
    A = np.ones((len(x), 2))
    r = OptimizeResult(
        gp=gp,
        status=-1,
        success=False,
        weights=[0.0, 1.0],
        lnlike=-np.inf,
        fun=np.inf,
        x=None,
        prediction=None,
        message="initialized but failed",
    )
    for pars in parameters:
        mod = model(pars, x)
        if mod is None:
            continue
        A[:, 0] = mod
        ATA = np.dot(A.T, gp.solver.apply_inverse(A))
        ATy = np.dot(A.T, gp.solver.apply_inverse(y))
        w = np.linalg.solve(ATA, ATy)
        mod = np.dot(A, w)

        # Compute the likelihood.
        ll = gp.lnlikelihood(y - mod, quiet=True)
        if ll > r.lnlike:
            r.lnlike = ll
            r.status = 0
            r.success = True
            r.fun = -ll
            r.weights = w
            r.x = pars
            r.prediction = mod
            r.message = "success"
    if r.prediction is not None:
        bkg = gp.predict(y - r.prediction, x, mean_only=True)
        r.prediction += bkg
    return r


def fit_transit(x, y, yerr, t0, gp=None, nrestart=10, delta_t0=0.2):
    if gp is None:
        gp = init_gp(x, y, yerr)

    # Remove missing data.
    m = np.isfinite(y) & np.isfinite(y)
    xm = x[m]
    ym = y[m]

    # Define the model.
    def nll(p):
        system = get_system(p)
        r = ym - system.light_curve(xm)
        return -gp.lnlikelihood(r, quiet=True)

    # Optimize.
    t0rng = (max(xm.min(), t0-delta_t0), min(xm.max(), t0+delta_t0))
    bounds = [(-5.0, 0.0), (5.0, 9.0), t0rng, (0.0, 1.0)]
    result = None
    for i in range(nrestart):
        p0 = [np.random.uniform(*r) for r in bounds]
        r = minimize(nll, p0, method="L-BFGS-B", bounds=bounds)
        if result is None or result.fun >= r.fun:
            result = r

    # Save the results.
    result.system = get_system(result.x)
    result.gp = gp
    result.lnlike = -result.fun

    # Compute prediction.
    mod = result.system.light_curve(xm)
    bkg = result.gp.predict(ym - mod, xm, mean_only=True)
    result.prediction = mod + bkg
    return result


def get_system(p):
    r, period = np.exp(p[:2])
    t0, b = p[2:]
    system = transit.System(transit.Central())
    system.add_body(transit.Body(r=r, period=period, t0=t0, b=b))
    return system


def init_gp(x, y, yerr):
    tau = 0.25 * estimate_tau(x, y)
    kernel = np.median((y - np.median(y))**2) * Matern32Kernel(tau*tau)
    gp = george.GP(kernel)
    gp.compute(x, yerr)
    return gp


def acor_fn(x):
    """Compute the autocorrelation function of a time series."""
    n = len(x)
    f = np.fft.fft(x-np.mean(x), n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:n].real
    return acf / acf[0]


def estimate_tau(t, y):
    """Estimate the correlation length of a time series."""
    dt = np.min(np.diff(t))
    tt = np.arange(t.min(), t.max(), dt)
    yy = np.interp(tt, t, y, 1)
    f = acor_fn(yy)
    fs = gaussian_filter(f, 50)
    w = dt * np.arange(len(f))
    m = np.arange(1, len(fs)-1)[(fs[1:-1] > fs[2:]) & (fs[1:-1] > fs[:-2])]
    if len(m):
        return w[m[np.argmax(fs[m])]]
    return w[-1]
