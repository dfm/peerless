# -*- coding: utf-8 -*-

from __future__ import division, print_function

from plot_setup import COLORS
from matplotlib import rcParams
rcParams["font.size"] = 16

import h5py
import emcee
import transit
import peerless
import numpy as np
import pandas as pd
from scipy.stats import beta
import matplotlib.pyplot as pl

kicid = 10602068

# Load the data.
kic = peerless.catalogs.KICatalog().df
cands = pd.read_csv("3k_candidates.csv")
cands = pd.merge(cands, kic, left_on="kicid", right_on="kepid", how="left",
                 suffixes=["", "_kic"])

# Pull out the target candidate.
cand = cands[cands.kicid == kicid]
period = float(cand.nn_period)
rp = float(cand.nn_rp)
b = float(cand.nn_b)
t0 = float(cand.time)

rstar = float(cand.radius)
rstar_err1 = float(cand.radius_err1)
rstar_err2 = float(cand.radius_err2)
ln_rstar = np.log(rstar)
ln_rstar_err = max(np.log(rstar + rstar_err1) - np.log(rstar),
                   np.log(rstar) - np.log(rstar + rstar_err2))

mstar = float(cand.mass)
mstar_err1 = float(cand.mass_err1)
mstar_err2 = float(cand.mass_err2)
ln_mstar = np.log(mstar)
ln_mstar_err = max(np.log(mstar + mstar_err1) - np.log(mstar),
                   np.log(mstar) - np.log(mstar + mstar_err2))

# Set up the model.
system = transit.System(transit.Central(mass=mstar, radius=rstar))
system.add_body(transit.Body(
    period=period, r=1.3*rp, b=b, t0=t0,
))

# Load the light curves.
lcs = peerless.data.load_light_curves_for_kic(kicid, min_break=10)
texp = lcs[0].texp
t = np.concatenate([lc.time for lc in lcs])
f = np.concatenate([lc.flux for lc in lcs])
fe = np.concatenate([lc.yerr + np.zeros_like(lc.time) for lc in lcs])

# Find the minimum allowed period.
m = np.isfinite(t)
min_period = max(t[m].max() - t0, t0 - t[m].min())

# Only select data near the transit.
m = (np.abs(t - t0) < 6) & np.isfinite(t) & np.isfinite(f)
t, f, fe = map(np.ascontiguousarray, (t[m], f[m], fe[m]))


# Set up the emcee model.
class TransitWalker(emcee.BaseWalker):

    def lnpriorfn(self, p):
        lnfs, lnrs, lnms, q1, q2 = p[:5]
        lp = 0.0
        if not ((0 < q1 < 1) and (0 < q2 < 1)):
            return -np.inf
        lp -= 0.5 * (((lnrs - ln_rstar) / ln_rstar_err) ** 2 +
                     ((lnms - ln_mstar) / ln_mstar_err) ** 2)

        lnr, lnp, t0, b, sesn, secs = p[5:]
        if not 0 <= b < 2.0:
            return -np.inf
        if np.exp(lnp) < min_period:
            return -np.inf
        e = sesn**2 + secs**2
        if not 0 <= e < 1.0:
            return -np.inf

        lp += beta.logpdf(e, 1.12, 3.09)
        return lp

    def lnlikefn(self, p):
        lnfs, lnrs, lnms, q1, q2 = p[:5]
        lnr, lnp, t0, b, sesn, secs = p[5:]

        star = system.central
        star.mass = np.exp(lnms)
        star.radius = np.exp(lnrs)
        star.q1, star.q2 = q1, q2

        planet = system.bodies[0]
        e = sesn**2 + secs**2
        pomega = np.arctan2(sesn, secs)

        planet.period = np.exp(lnp)
        planet.r = np.exp(lnr)
        planet.t0 = t0
        planet.e = e
        planet.pomega = pomega
        try:
            planet.b = b
        except ValueError:
            return -np.inf

        pred = np.exp(lnfs) * system.light_curve(t, texp=texp)
        ll = -0.5 * np.sum(((f - pred) / fe)**2)
        return ll

p0 = np.array([0.0, ln_rstar, ln_mstar, 0.5, 0.5, np.log(rp), np.log(period),
               t0, b, 0.0, 0.0])
nwalkers, ndim = 32, len(p0)
coords = p0 + 1e-8 * np.random.randn(nwalkers, ndim)
ensemble = emcee.Ensemble(TransitWalker(), coords)
assert np.all(np.isfinite(ensemble.lnprob))
sampler = emcee.Sampler()

print("Burn-in 1")
ensemble = sampler.run(ensemble, 2000)

# Re-sample ensemble.
samples = sampler.get_coords(flat=True)
lp = sampler.get_lnprob(flat=True)
best_p = samples[np.argmax(lp)]
print(best_p)
print(np.exp(best_p))
coords = p0 + 1e-8 * np.random.randn(nwalkers, ndim)
ensemble = emcee.Ensemble(TransitWalker(), coords)

print("Burn-in 2")
sampler.reset()
ensemble = sampler.run(ensemble, 5000)

print("Production")
sampler.reset()
ensemble = sampler.run(ensemble, 50000)

with h5py.File("params.h5", "w") as fh:
    fh.create_dataset("coords", data=sampler.get_coords())
    fh.create_dataset("lnprob", data=sampler.get_lnprob())

# Find best sample and plot.
samples = sampler.get_coords(flat=True)
lp = sampler.get_lnprob(flat=True)
best_p = samples[np.argmax(lp)]
TransitWalker().lnpriorfn(best_p)

pl.plot(t, f, ".k", ms=2)
pl.plot(t, np.exp(best_p[0]) * system.light_curve(t, texp=lcs[0].texp))
pl.savefig("params.pdf")
