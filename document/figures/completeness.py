# -*- coding: utf-8 -*-

from __future__ import division, print_function

from plot_setup import COLORS, SQUARE_FIGSIZE

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

inj = pd.read_hdf("../../results/injections.h5", "injections")
inj = inj[inj.ncadences > 0]
rec = inj[inj.recovered]

# n_all, bins = np.histogram(np.log10(inj.radius), 20)
# n_rec, _ = np.histogram(np.log10(rec.radius), bins)
# n = n_rec / n_all
# n_err = n / np.sqrt(n_all)
# print(n)

n_all, ln_radius_bins, ln_period_bins = np.histogram2d(
    np.log(inj.radius), np.log(inj.period), (10, 5),
)
n_rec, _, _ = np.histogram2d(
    np.log(rec.radius), np.log(rec.period), (ln_radius_bins, ln_period_bins),
)
n = n_rec / n_all
n_err = n / np.sqrt(n_all)

X, Y = np.meshgrid(np.exp(ln_period_bins) / 365.25,
                   np.exp(ln_radius_bins) / 0.0995,
                   indexing="ij")

fig, ax = pl.subplots(1, 1, figsize=SQUARE_FIGSIZE)
cax = ax.pcolor(X, Y, 100*n.T, cmap="viridis", vmin=0, vmax=100)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_xlabel("period [years]")
ax.set_ylabel("$R_\mathrm{P} / R_\mathrm{J}$")
ax.set_xticks([3, 5, 10, 20])
ax.get_xaxis().set_major_formatter(pl.ScalarFormatter())
ax.set_yticks([0.1, 0.2, 0.5, 1, 2])
ax.get_yaxis().set_major_formatter(pl.ScalarFormatter())

fig.savefig("completeness.pdf", bbox_inches="tight")
