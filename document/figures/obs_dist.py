# -*- coding: utf-8 -*-

from peerless.plot_setup import COLORS, SQUARE_FIGSIZE

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

from peerless.catalogs import TargetCatalog, KOICatalog

targets = TargetCatalog().df
kois = pd.merge(KOICatalog().df, targets, how="inner", on="kepid")
kois = kois[(kois.koi_pdisposition == "CANDIDATE") &
            np.isfinite(kois.koi_prad) &
            (200 < kois.koi_period) &
            (1 < kois.koi_prad) & (kois.koi_prad < 10)]

fits = pd.read_csv("../../results/fits.csv")
fits = fits[fits.period < 5000 / 365.25]

s = np.array(SQUARE_FIGSIZE)
s[0] *= 2
fig, axes = pl.subplots(1, 2, figsize=s)

def make_plot(ax, x1, x2, rng, nbins, ticks):
    bins = np.exp(np.linspace(np.log(rng[0]), np.log(rng[1]), nbins))

    ax.hist([x2, x1], bins, histtype="stepfilled", stacked=False,
            color=[COLORS["MODEL_1"], COLORS["MODEL_2"]],
            alpha=0.1, lw=0)
    n, _, _ = ax.hist(x1, bins, histtype="step", color=COLORS["MODEL_2"], lw=2)
    ax.hist(x2, bins, histtype="step", color=COLORS["MODEL_1"], lw=2,
            hatch="/")
    ax.set_ylim(0, np.max(n) + 0.5)

    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(pl.ScalarFormatter())
    ax.set_xlim(rng)

make_plot(axes[0], np.array(kois.koi_period), np.array(fits.period) * 365.25,
          (2e2, 5e3), 7, (200, 500, 1000, 2000, 5000))
make_plot(axes[1], np.array(kois.koi_prad), np.array(fits.radius) / 0.092051,
          (1, 10), 7, (1, 2, 5, 10))

axes[0].set_ylabel("observed number")
axes[0].set_xlabel("orbital period [days]")
axes[1].set_xlabel("planet radius [$R_\oplus$]")

fig.savefig("obs_dist.pdf", bbox_inches="tight")
