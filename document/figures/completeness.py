# -*- coding: utf-8 -*-

from __future__ import division, print_function

from plot_setup import SQUARE_FIGSIZE, COLORS

import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as pl

inj = pd.read_hdf("../../results/injections.h5", "injections")
inj = inj[inj.ncadences > 0]
rec = inj[inj.recovered]

n_all, ln_radius_bins, ln_period_bins = np.histogram2d(
    np.log(inj.radius), np.log(inj.period), (8, 4),
)
n_rec, _, _ = np.histogram2d(
    np.log(rec.radius), np.log(rec.period), (ln_radius_bins, ln_period_bins),
)
n = n_rec / n_all
n_err = n / np.sqrt(n_all)

X, Y = np.meshgrid(np.exp(ln_period_bins) / 365.25,
                   np.exp(ln_radius_bins) / 0.0995,
                   indexing="ij")

fig = pl.figure(figsize=2*np.array(SQUARE_FIGSIZE))

ax = pl.axes([0.1, 0.1, 0.6, 0.6])
ax.pcolor(X, Y, 100*n.T, cmap="viridis", vmin=0, vmax=100)

# Label the bins with their completeness percentages.
for i, j in product(range(len(ln_period_bins)-1),
                    range(len(ln_radius_bins)-1)):
    x = np.exp(0.5 * (ln_period_bins[i] + ln_period_bins[i+1])) / 365.25
    y = np.exp(0.5 * (ln_radius_bins[j] + ln_radius_bins[j+1])) / 0.0995
    ax.annotate(r"${0:.1f} \pm {1:.1f}$".format(100*n[j, i], 100*n_err[j, i]),
                (x, y), ha="center", va="center", alpha=1.0, fontsize=12,
                color="white")

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

# Histograms.

# Top:
ax = pl.axes([0.1, 0.71, 0.6, 0.15])
x = np.exp(ln_period_bins) / 365.25
y = 100*(
    np.histogram(np.log(rec.period), ln_period_bins)[0] /
    np.histogram(np.log(inj.period), ln_period_bins)[0]
)
x = np.array(list(zip(x[:-1], x[1:]))).flatten()
y = np.array(list(zip(y, y))).flatten()
ax.plot(x, y, lw=1, color=COLORS["DATA"])
ax.fill_between(x, y, np.zeros_like(y), color=COLORS["DATA"], alpha=0.2)
ax.set_xlim(X.min(), X.max())
ax.set_ylim(0, 80)
ax.set_xscale("log")
ax.set_xticks([3, 5, 10, 20])
ax.set_xticklabels([])
ax.yaxis.set_major_locator(pl.MaxNLocator(3))

# Right:
ax = pl.axes([0.71, 0.1, 0.15, 0.6])
x = np.exp(ln_radius_bins) / 0.0995
y = 100*(
    np.histogram(np.log(rec.radius), ln_radius_bins)[0] /
    np.histogram(np.log(inj.radius), ln_radius_bins)[0]
)
x = np.array(list(zip(x[:-1], x[1:]))).flatten()
y = np.array(list(zip(y, y))).flatten()
ax.plot(y, x, lw=1, color=COLORS["DATA"])
ax.fill_betweenx(x, y, np.zeros_like(y), color=COLORS["DATA"], alpha=0.2)
ax.set_ylim(Y.min(), Y.max())
ax.set_xlim(0, 80)
ax.set_yscale("log")
ax.set_yticks([0.1, 0.2, 0.5, 1, 2])
ax.set_yticklabels([])
ax.xaxis.set_major_locator(pl.MaxNLocator(3))

fig.savefig("completeness.pdf", bbox_inches="tight")
