# -*- coding: utf-8 -*-

from __future__ import division, print_function

from plot_setup import COLORS, SQUARE_FIGSIZE

import peerless
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.ticker import ScalarFormatter

# Newton's constant in R_Sun^3 M_Sun^{-1} {days}^{-2}.
G = 2945.4625385377644

# Get the stellar data.
kic = peerless.catalogs.KICatalog().df

# This is the factor out front of the integral.
T_k = kic.dataspan*kic.dutycycle
factor = np.array(T_k * kic.radius * (4*np.pi**2/(G*kic.mass))**(1./3))

# Pre-compute the stellar noise and base radius scale.
sig_k = 1e-6 * kic.rrmscdpp15p0
r_k = np.array(np.sqrt(sig_k) * kic.radius)

# We'll integrate for all periods above 1500 days.
P_min = 1500.

# These are the fits in radial bins from Dong & Zhu (2013)
rad_rngs = [
    # R_min, R_max, C, C_err, beta, beta_err
    (0.01, 0.02, 0.66, 0.08, -0.10, 0.12),
    (0.02, 0.04, 0.49, 0.03, 0.11, 0.05),
    (0.04, 0.08, 0.040, 0.008, 0.70, 0.10),
    (0.08, 0.16, 0.023, 0.007, 0.50, 0.17),
]


# Compute the expected number of single transits for a given detection
# threshold.
def compute_expected_number(threshold):
    N = np.zeros((3, len(factor)))
    for rmn, rmx, C, C_err, beta, beta_err in rad_rngs:
        for j, (bb, cc) in enumerate(zip(beta + beta_err*np.array([-1, 0, 1]),
                                         C + C_err*np.array([-1, 0, 1]))):
            # Compute the bounds of the radius integral.
            rmin = np.array(np.sqrt(threshold) * r_k)
            rmin[rmin < rmn] = rmn
            rmin[rmin > rmx] = rmx

            # Compute the integral over period.
            p = bb - 5./3
            p_fac = -P_min**p / p

            # Convert the Dong & Zhu rate to natural logarithms.
            dong_fac = cc / (np.log(10) * 10**bb)

            N[j] += p_fac * dong_fac * (np.log(rmx) - np.log(rmin))

    m = np.isfinite(factor)
    return (factor[None, m] * N[:, m]).sum(axis=1)

q = compute_expected_number(15)
print(q[1], np.diff(q))

fs = np.exp(np.linspace(np.log(1), np.log(100), 20))
N = np.array(map(compute_expected_number, fs))

fig, ax = pl.subplots(1, 1, figsize=SQUARE_FIGSIZE)
color = COLORS["DATA"]
ax.fill_between(fs, N[:, 0], N[:, 2], color=color, alpha=0.2)
ax.plot(fs, N[:, 0], color=color, lw=0.5)
ax.plot(fs, N[:, 2], color=color, lw=0.5)
ax.set_xlim(5, 100)
ax.set_ylim(10, 400)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("$f$")
ax.set_ylabel("$N$")

ax.set_yticks([10, 30, 100, 300])
ax.get_yaxis().set_major_formatter(ScalarFormatter())
ax.set_xticks([10, 100])
ax.get_xaxis().set_major_formatter(ScalarFormatter())

fig.savefig("predict.pdf")
