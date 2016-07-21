# -*- coding: utf-8 -*-

from peerless.plot_setup import COLORS, SQUARE_FIGSIZE

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

from peerless.catalogs import TargetCatalog, KOICatalog

targets = TargetCatalog().df
kois = pd.merge(KOICatalog().df, targets, how="inner")
print(kois)
assert 0

fits = pd.read_csv("../../results/fits.csv")

fig, ax = pl.subplots(1, 1, figsize=SQUARE_FIGSIZE)

x = np.array(fits.radius)
bins = np.exp(np.linspace(np.log(0.1), np.log(2.0), 6))
bins[-1] = 2.01
n, _, _ = ax.hist(x, bins, histtype="step", color=COLORS["DATA"])
bins[-1] = 2.0
ax.set_xscale("log")
ax.set_xticks([0.1, 0.2, 0.5, 1, 2])
ax.get_xaxis().set_major_formatter(pl.ScalarFormatter())
ax.set_xlim(0.1, 2)
ax.set_ylim(0, 6.5)
ax.set_ylabel("observed number")
ax.set_xlabel("planet radius [$R_\mathrm{J}$]")

fig.savefig("radius_dist.pdf", bbox_inches="tight")
