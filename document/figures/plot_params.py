# -*- coding: utf-8 -*-

from __future__ import division, print_function

from plot_setup import COLORS
from matplotlib import rcParams
rcParams["font.size"] = 16

import h5py
import transit
import triangle
import peerless
import numpy as np
import pandas as pd
from scipy.stats import beta
import matplotlib.pyplot as pl


with h5py.File("params.h5", "r") as fh:
    samples = fh["coords"][...]
    lnprob = fh["lnprob"][...]

pl.clf()
pl.plot(samples[:, :, 1], alpha=0.5)
pl.savefig("time.png")


flat = samples[::10].reshape((-1, samples.shape[-1]))

e = np.sum(flat[:, -2:]**2, axis=1)
pl.clf()
pl.hist(e, 40, normed=True)
x = np.linspace(1e-5, 0.8, 500)
pl.plot(x, beta.pdf(x, 1.12, 3.09))
pl.savefig("blah.png")


r_star = np.exp(flat[:, 1])
r_p = np.exp(flat[:, 5]) / 0.0995

fig = triangle.corner(np.vstack([r_p, np.log10(np.exp(flat[:, 6])),
                                 flat[:, 8], e]).T,
                      levels=[0.68, 0.95],
                      smooth=1.0, smooth1d=1.0,
                      fill_contours=True, plot_datapoints=False,
                      labels=["$R_p / R_\mathrm{J}$",
                              r"$\log_{10}\,P/\mathrm{day}$", "$b$", "$e$"])
fig.savefig("corner.pdf")

assert 0

# Compute the radius.
r_p = flat[:, 1]  # np.exp(flat[:, 5]) / 0.0995
pl.hist(r_p)
pl.savefig("blah.png")
q = triangle.quantile(r_p, [0.16, 0.5, 0.84])
print(q[0], np.diff(q))

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
print(ln_rstar, ln_rstar_err)
assert 0

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


# Compute the predicted light curve for a given set of parameters.
def prediction(p):
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

    return np.exp(lnfs) * system.light_curve(t, texp=texp)


i, j = np.unravel_index(np.argmax(lnprob), lnprob.shape)
best_p = samples[i, j]

pl.plot(t, f, ".", color=COLORS["DATA"])
pl.plot(t, prediction(best_p), "g")
pl.savefig("params.pdf")
