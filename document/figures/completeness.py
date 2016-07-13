# -*- coding: utf-8 -*-

from __future__ import division, print_function

from peerless.plot_setup import SQUARE_FIGSIZE, COLORS

# import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as pl
from scipy.optimize import minimize

from autograd import grad
import autograd.numpy as np


inj = pd.read_hdf("../../results/injections-with-mass.h5", "injections")
inj = inj[inj.ncadences > 0]
rec = inj[inj.recovered]


def analytic_model(params, ln_radius, ln_period):
    b1, b2, a1, a2, k1, k2, x1, x2 = params
    a = (a1 * ln_period + a2) * (b1 * ln_radius + b2)
    a = a * (a > 0.0) * (a < 1.0) + 1.0 * (a > 1.0)
    k = k1 * ln_period + k2
    x = x1 * ln_period + x2
    return a / (1.0 + np.exp(-k * (ln_radius - x)))


def nll(params, ln_radius, ln_period, flag):
    m = analytic_model(params, ln_radius, ln_period)
    return np.mean((flag - m)**2)


def error(params, ln_radius, ln_period, flag):
    m = analytic_model(params, ln_radius, ln_period)
    pred = m > 0.5
    return np.sum(pred != flag)


x, y = np.log(np.array(inj.radius)), np.log(np.array(inj.period))
flag = np.array(inj.recovered, dtype=int)
params = np.array([0.0, 1.0, 0.0, 0.6, 0.0, 10.0, 0.0, -3.0])
r = minimize(nll, params, jac=grad(nll), args=(x, y, flag))
with open("../completenessfit.tex", "w") as f:
    f.write(r"""\newcommand{{\parama}}{{{0[0]:.2f}}}
\newcommand{{\paramb}}{{{0[1]:.2f}}}
\newcommand{{\paramc}}{{{0[2]:.2f}}}
\newcommand{{\paramd}}{{{0[3]:.2f}}}
\newcommand{{\parame}}{{{0[4]:.2f}}}
\newcommand{{\paramf}}{{{0[5]:.2f}}}
\newcommand{{\paramg}}{{{0[6]:.2f}}}
\newcommand{{\paramh}}{{{0[7]:.2f}}}
""".format(r.x))

# Plot - analytic
s = np.array(SQUARE_FIGSIZE)
s[0] *= 1.25
fig, ax = pl.subplots(1, 1, figsize=s)

ln_radius_bins = np.linspace(np.log(0.15), np.log(2.2), 100)
ln_period_bins = np.linspace(np.log(2), np.log(25), 101)
Y, X = np.meshgrid(ln_period_bins, ln_radius_bins, indexing="ij")
Z = analytic_model(r.x, X + np.log(0.0995), Y + np.log(365.25))
c = ax.pcolor(np.exp(ln_period_bins), np.exp(ln_radius_bins), Z.T,
              cmap="viridis", vmin=0, vmax=1)
cbar = fig.colorbar(c)
cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.contour(np.exp(Y), np.exp(X), Z, [0.1, 0.3, 0.5, 0.7], colors="w")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(np.exp([X.min(), X.max()]))
ax.set_xlim(np.exp([Y.min(), Y.max()]))
ax.set_xlabel("period [years]")
ax.set_ylabel("$R_\mathrm{P} / R_\mathrm{J}$")
ax.set_xticks([3, 5, 10, 20])
ax.get_xaxis().set_major_formatter(pl.ScalarFormatter())
ax.set_yticks([0.2, 0.5, 1, 2])
ax.get_yaxis().set_major_formatter(pl.ScalarFormatter())
fig.savefig("completeness_analytic.pdf", bbox_inches="tight")


# Histograms.
ln_radius_bins = np.linspace(np.log(0.15), np.log(2.2), 8) + np.log(0.0995)
ln_period_bins = np.linspace(np.log(2), np.log(25), 5) + np.log(365.25)
n_all, ln_radius_bins, ln_period_bins = np.histogram2d(
    np.log(inj.radius), np.log(inj.period), (ln_radius_bins, ln_period_bins),
)
n_rec, _, _ = np.histogram2d(
    np.log(rec.radius), np.log(rec.period), (ln_radius_bins, ln_period_bins),
)
n = n_rec / n_all
n_err = n / np.sqrt(n_all)

X, Y = np.meshgrid(np.exp(ln_period_bins) / 365.25,
                   np.exp(ln_radius_bins) / 0.0995,
                   indexing="ij")


# Plot 2
fig = pl.figure(figsize=2*np.array(SQUARE_FIGSIZE))

ax = pl.axes([0.1, 0.1, 0.6, 0.6])
ax.pcolor(X, Y, 100*n.T, cmap="viridis", vmin=0, vmax=100)

# Label the bins with their completeness percentages.
for i, j in product(range(len(ln_period_bins)-1),
                    range(len(ln_radius_bins)-1)):
    x = np.exp(0.5 * (ln_period_bins[i] + ln_period_bins[i+1])) / 365.25
    y = np.exp(0.5 * (ln_radius_bins[j] + ln_radius_bins[j+1])) / 0.0995
    ax.annotate(r"${0:.3f}$".format(n[j, i]),
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
ax.set_yticks([0.2, 0.5, 1, 2])
ax.get_yaxis().set_major_formatter(pl.ScalarFormatter())

# Histograms.

# Top:
ax = pl.axes([0.1, 0.71, 0.6, 0.15])
x = np.exp(ln_period_bins) / 365.25
y = (
    np.histogram(np.log(rec.period), ln_period_bins)[0] /
    np.histogram(np.log(inj.period), ln_period_bins)[0]
)
x = np.array(list(zip(x[:-1], x[1:]))).flatten()
y = np.array(list(zip(y, y))).flatten()
ax.plot(x, y, lw=1, color=COLORS["DATA"])
ax.fill_between(x, y, np.zeros_like(y), color=COLORS["DATA"], alpha=0.2)
ax.set_xlim(X.min(), X.max())
ax.set_ylim(0, 0.8)
ax.set_xscale("log")
ax.set_xticks([3, 5, 10, 20])
ax.set_xticklabels([])
ax.yaxis.set_major_locator(pl.MaxNLocator(3))

# Right:
ax = pl.axes([0.71, 0.1, 0.15, 0.6])
x = np.exp(ln_radius_bins) / 0.0995
y = (
    np.histogram(np.log(rec.radius), ln_radius_bins)[0] /
    np.histogram(np.log(inj.radius), ln_radius_bins)[0]
)
x = np.array(list(zip(x[:-1], x[1:]))).flatten()
y = np.array(list(zip(y, y))).flatten()
ax.plot(y, x, lw=1, color=COLORS["DATA"])
ax.fill_betweenx(x, y, np.zeros_like(y), color=COLORS["DATA"], alpha=0.2)
ax.set_ylim(Y.min(), Y.max())
ax.set_xlim(0, 0.8)
ax.set_yscale("log")
ax.set_yticks([0.2, 0.5, 1, 2])
ax.set_yticklabels([])
ax.xaxis.set_major_locator(pl.MaxNLocator(3))

fig.savefig("completeness.pdf", bbox_inches="tight")
pl.close(fig)


# Histograms.
ln_mass_bins = np.linspace(np.log(2.0), np.log(1e5), 8)
ln_semimajor_bins = np.linspace(np.log(1.5), np.log(9), 5) + np.log(215.1)
n_all, ln_mass_bins, ln_semimajor_bins = np.histogram2d(
    inj.log10_mass*np.log(10), np.log(inj.semimajor),
    (ln_mass_bins, ln_semimajor_bins),
)
n_rec, _, _ = np.histogram2d(
    rec.log10_mass*np.log(10), np.log(rec.semimajor),
    (ln_mass_bins, ln_semimajor_bins),
)
n = n_rec / n_all
n_err = n / np.sqrt(n_all)

X, Y = np.meshgrid(np.exp(ln_semimajor_bins) / 215.1,
                   np.exp(ln_mass_bins) / 317.828,
                   indexing="ij")


# Plot 2
fig = pl.figure(figsize=2*np.array(SQUARE_FIGSIZE))

ax = pl.axes([0.1, 0.1, 0.6, 0.6])
ax.pcolor(X, Y, 100*n.T, cmap="viridis", vmin=0, vmax=100)

# Label the bins with their completeness percentages.
for i, j in product(range(len(ln_semimajor_bins)-1),
                    range(len(ln_mass_bins)-1)):
    x = np.exp(0.5 * (ln_semimajor_bins[i] + ln_semimajor_bins[i+1])) / 215.1
    y = np.exp(0.5 * (ln_mass_bins[j] + ln_mass_bins[j+1])) / 317.828
    ax.annotate(r"${0:.3f}$".format(n[j, i]),
                (x, y), ha="center", va="center", alpha=1.0, fontsize=12,
                color="white")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_xlabel("semi-major axis [AU]")
ax.set_ylabel("$M_\mathrm{P} / M_\mathrm{J}$")
ax.set_xticks([2, 4, 8])
ax.get_xaxis().set_major_formatter(pl.ScalarFormatter())
ax.set_yticks([0.01, 0.1, 1, 10, 100])
ax.set_yticklabels(["0.01", "0.1", "1", "10", "100"])

# Histograms.

# Top:
ax = pl.axes([0.1, 0.71, 0.6, 0.15])
x = np.exp(ln_semimajor_bins) / 215.1
y = (
    np.histogram(np.log(rec.semimajor), ln_semimajor_bins)[0] /
    np.histogram(np.log(inj.semimajor), ln_semimajor_bins)[0]
)
x = np.array(list(zip(x[:-1], x[1:]))).flatten()
y = np.array(list(zip(y, y))).flatten()
ax.plot(x, y, lw=1, color=COLORS["DATA"])
ax.fill_between(x, y, np.zeros_like(y), color=COLORS["DATA"], alpha=0.2)
ax.set_xlim(X.min(), X.max())
ax.set_ylim(0, 0.8)
ax.set_xscale("log")
ax.set_xticks([2, 4, 8])
ax.set_xticklabels([])
ax.yaxis.set_major_locator(pl.MaxNLocator(3))

# Right:
ax = pl.axes([0.71, 0.1, 0.15, 0.6])
x = np.exp(ln_mass_bins) / 317.828
y = (
    np.histogram(rec.log10_mass*np.log(10), ln_mass_bins)[0] /
    np.histogram(inj.log10_mass*np.log(10), ln_mass_bins)[0]
)
x = np.array(list(zip(x[:-1], x[1:]))).flatten()
y = np.array(list(zip(y, y))).flatten()
ax.plot(y, x, lw=1, color=COLORS["DATA"])
ax.fill_betweenx(x, y, np.zeros_like(y), color=COLORS["DATA"], alpha=0.2)
ax.set_ylim(Y.min(), Y.max())
ax.set_xlim(0, 0.8)
ax.set_yscale("log")
ax.set_yticks([0.01, 0.1, 1, 10, 100])
ax.set_yticklabels([])
ax.xaxis.set_major_locator(pl.MaxNLocator(3))

fig.savefig("completeness-am.pdf", bbox_inches="tight")
pl.close(fig)
