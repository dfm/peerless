# -*- coding: utf-8 -*-

__all__ = ["TransitModel", "setup_fit"]

import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as pl
from scipy.optimize import minimize

import george
from george import kernels

import transit

from .catalogs import KOICatalog
from .data import load_light_curves_for_kic


class TransitModel(object):

    eb = beta(1.12, 3.09)

    def __init__(self, spec, gps, system, smass, smass_err, srad, srad_err,
                 fit_lcs, other_lcs):
        self.spec = spec

        # Put a prior range on the reference time.
        t0 = system.bodies[0].t0
        self.t0rng = t0 + np.array([-0.5, 0.5])

        # Put a minimum prior bound on the period.
        tmn = np.min([np.min(lc.time) for lc in np.append(other_lcs, fit_lcs)])
        tmx = np.max([np.max(lc.time) for lc in np.append(other_lcs, fit_lcs)])
        self.min_period = np.max([np.abs(t0 - tmn), np.abs(tmx - t0)])

        self.gps = gps
        self.system = system
        self.smass = smass
        self.smass_err = smass_err
        self.srad = srad
        self.srad_err = srad_err
        self.fit_lcs = fit_lcs
        self.other_lcs = other_lcs

        body = self.system.bodies[0]
        mx = (
            self.lnlike(compute_blob=False)[0],
            (body.b, body.radius, body.period, body.e, body.omega)
        )
        prng = np.exp(np.append(np.linspace(np.log(0.5), np.log(2.0), 6), 0))
        rrng = body.radius * np.append(np.linspace(0.8, 1.2, 6), 1.0)
        rstar = self.system.central.radius
        for ecc in np.linspace(0, 0.8, 4):
            body.e = ecc
            for w in np.linspace(-np.pi, np.pi, 4):
                body.omega = w
                for per in body.period*prng:
                    body.period = per
                    for rad in rrng:
                        body.radius = rad
                        for b in np.linspace(0, 1.0+0.9*rad/rstar, 6):
                            body.b = b
                            ll, blob = self.lnlike(compute_blob=False)
                            if ll > mx[0]:
                                ll, blob = self.lnlike(compute_blob=True)
                                if blob[0] == 0:
                                    mx = (ll, (b, rad, per, ecc, w))
        body.e = mx[1][3]
        body.omega = mx[1][4]
        body.period = mx[1][2]
        body.b = mx[1][0]
        body.radius = mx[1][1]

    # Probabilistic model:
    def lnprior(self):
        if not (self.t0rng[0] < self.system.bodies[0].t0 < self.t0rng[1]):
            return -np.inf

        star = self.system.central
        if np.any([b.radius > star.radius for b in self.system.bodies]):
            return -np.inf

        lp = 0.0

        # planet
        body = self.system.bodies[0]

        # Minimum period.
        if body.period < self.min_period:
            return -np.inf

        # Impact parameter.
        if body.b < 0.0:
            return -np.inf
        elif body.b > 2.0:
            return -np.inf
        elif body.b > 1.0:
            lp += np.log(np.cos(0.5*np.pi*(1.0 - body.b)))
            # lp -= body.b - 1.0

        # Flat in log a (and period)
        lp -= np.log(body.a)

        # Beta in eccen
        lp += self.eb.logpdf(body.e)

        # stellar parameters
        lp -= 0.5 * (
            ((star.mass - self.smass) / self.smass_err) ** 2 +
            ((star.radius - self.srad) / self.srad_err) ** 2
        )

        # limb darkening etc.
        lp += self.system.jacobian()

        return lp

    def lnlike(self, compute_blob=True):
        system = self.system
        ll = 0.0
        for gp, lc in zip(self.gps, self.fit_lcs):
            mu = system.light_curve(lc.time, texp=lc.texp, maxdepth=2)
            r = (lc.flux - mu) * 1e3
            ll += gp.lnlikelihood(r, quiet=True)
            if not (np.any(mu < system.central.flux) and np.isfinite(ll)):
                return -np.inf, (0, 0.0)

        y = system.light_curve(system.bodies[0].t0, texp=lc.texp, maxdepth=2)
        f = system.central.flux
        depth = float((f - y) / f)

        if not compute_blob:
            return ll, (0, depth)

        # # Compute number of cadences with transits in the other light curves.
        # ncad = sum((system.light_curve(lc.time) < system.central.flux).sum()
        #            for lc in self.other_lcs)
        ncad = 0

        return ll, (ncad, depth)

    def _update_params(self, theta):
        self.system.set_vector(theta[:len(self.system)])

    def lnprob(self, theta, compute_blob=True):
        blob = [None, 0, 0.0, 0.0]
        try:
            self._update_params(theta)
        except ValueError:
            return -np.inf, blob

        blob[0] = (
            self.system.bodies[0].period,
            self.system.bodies[0].e,
            self.system.bodies[0].b,
        )

        blob[3] = lp = self.lnprior()
        if not np.isfinite(lp):
            return -np.inf, blob

        ll, (blob[1], blob[2]) = self.lnlike(compute_blob=compute_blob)
        if not np.isfinite(ll):
            return -np.inf, blob

        # reject samples with other transits.
        if blob[1] > 0:
            return -np.inf, blob

        return lp + ll, blob

    def __call__(self, theta):
        return self.lnprob(theta)

    def optimize(self, niter=3):
        self.system.freeze_parameter("central:*")
        self.system.freeze_parameter("bodies*t0")
        p0 = self.system.get_vector()
        r = minimize(self._nll, p0, jac=self._grad_nll, method="L-BFGS-B")
        if r.success:
            self.system.set_vector(r.x)
        else:
            self.system.set_vector(p0)
        self.system.bodies[0].b = np.abs(self.system.bodies[0].b)
        self.system.thaw_parameter("central:*")
        self.system.thaw_parameter("bodies*t0")

        if not niter > 1:
            self.system.freeze_parameter("central:*")
            p0 = self.system.get_vector()
            r = minimize(self._nll, p0, jac=self._grad_nll, method="L-BFGS-B")

            # Thaw the stellar parameters.
            self.system.thaw_parameter("central:*")
            self.system.freeze_parameter("central:dilution")

            # Final optimization.
            p0 = self.system.get_vector()
            r = minimize(self._nlp, p0, method="L-BFGS-B")

            return

        for gp, lc in zip(self.gps, self.fit_lcs):
            mu = self.system.light_curve(lc.time, texp=lc.texp, maxdepth=2)
            r = (lc.flux - mu) * 1e3
            p0 = gp.get_vector()
            r = minimize(gp.nll, p0, jac=gp.grad_nll, args=(r, ))
            if r.success:
                gp.set_vector(r.x)
            else:
                gp.set_vector(p0)
        self.optimize(niter=niter - 1)

    def _nlp(self, theta):
        lnp, blob = self.lnprob(theta)
        if blob[1] > 0 or not np.isfinite(lnp):
            return 1e10
        return -lnp

    def _nll(self, theta):
        try:
            self.system.set_vector(theta)
        except ValueError:
            return 1e10

        nll = 0.0
        system = self.system
        for gp, lc in zip(self.gps, self.fit_lcs):
            mu = system.light_curve(lc.time, texp=lc.texp, maxdepth=2)
            r = (lc.flux - mu) * 1e3
            nll -= gp.lnlikelihood(r, quiet=True)
            if not (np.any(mu < system.central.flux) and np.isfinite(nll)):
                return 1e10

        return nll

    def _grad_nll(self, theta):
        try:
            self.system.set_vector(theta)
        except ValueError:
            return np.zeros_like(theta)

        system = self.system
        g = np.zeros_like(theta)
        for gp, lc in zip(self.gps, self.fit_lcs):
            mu = system.light_curve(lc.time, texp=lc.texp, maxdepth=2)
            gmu = 1e3 * system.get_gradient(lc.time, texp=lc.texp, maxdepth=2)
            r = (lc.flux - mu) * 1e3
            alpha = gp.apply_inverse(r)
            g -= np.dot(gmu, alpha)
            if not (np.any(mu < system.central.flux)
                    and np.all(np.isfinite(g))):
                return np.zeros_like(theta)

        return g

    def plot(self):
        fig, ax = pl.subplots(1, 1)
        t0 = self.system.bodies[0].t0
        period = self.system.bodies[0].period
        for gp, lc in zip(self.gps, self.fit_lcs):
            t = (lc.time - t0 + 0.5*period) % period - 0.5*period
            mu = self.system.light_curve(lc.time, texp=lc.texp, maxdepth=2)
            r = (lc.flux - mu) * 1e3
            p = gp.predict(r, lc.time, return_cov=False) * 1e-3

            ax.plot(t, lc.flux, "k")
            ax.plot(t, p + self.system.central.flux, "g")
            ax.plot(t, mu, "r")
        return fig


def setup_fit(args, fit_kois=False, max_points=300):
    kicid = args["kicid"]

    # Initialize the system.
    system = transit.System(transit.Central(
        flux=1.0, radius=args["srad"], mass=args["smass"], q1=0.5, q2=0.5,
    ))
    system.add_body(transit.Body(
        radius=args["radius"],
        period=args["period"],
        t0=args["t0"],
        b=args["impact"],
        e=1.123e-7,
        omega=0.0,
    ))
    if fit_kois:
        kois = KOICatalog().df
        kois = kois[kois.kepid == kicid]
        for _, row in kois.iterrows():
            system.add_body(transit.Body(
                radius=float(row.koi_ror) * args["srad"],
                period=float(row.koi_period),
                t0=float(row.koi_time0bk) % float(row.koi_period),
                b=float(row.koi_impact),
                e=1.123e-7,
                omega=0.0,
            ))
    system.thaw_parameter("*")
    system.freeze_parameter("bodies*ln_mass")

    # Load the light curves.
    lcs, _ = load_light_curves_for_kic(kicid, remove_kois=not fit_kois)

    # Which light curves should be fit?
    fit_lcs = []
    other_lcs = []
    gps = []
    for lc in lcs:
        f = system.light_curve(lc.time, lc.texp)
        if np.any(f < 1.0):
            i = np.argmin(f)
            inds = np.arange(len(f))
            m = np.zeros(len(lc.time), dtype=bool)
            m[np.sort(np.argsort(np.abs(inds - i))[:max_points])] = True
            if np.any(f[~m] < system.central.flux):
                m = np.ones(len(lc.time), dtype=bool)
            lc.time = np.ascontiguousarray(lc.time[m])
            lc.flux = np.ascontiguousarray(lc.flux[m])
            lc.ferr = np.ascontiguousarray(lc.ferr[m])
            fit_lcs.append(lc)
            var = np.median((lc.flux - np.median(lc.flux))**2)
            kernel = 1e6*var * kernels.Matern32Kernel(2**2)
            gp = george.GP(kernel, white_noise=2*np.log(np.mean(lc.ferr)*1e3),
                           fit_white_noise=True)
            gp.compute(lc.time, lc.ferr * 1e3)
            gps.append(gp)
        else:
            other_lcs.append(lc)

    model = TransitModel(args, gps, system, args["smass"], args["smass_err"],
                         args["srad"], args["srad_err"], fit_lcs, other_lcs)
    return model
