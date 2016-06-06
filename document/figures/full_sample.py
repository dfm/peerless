# -*- coding: utf-8 -*-

from plot_setup import COLORS

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

from peerless.catalogs import KOICatalog

fig, ax = pl.subplots(1, 1, figsize=(6, 4))

kois = KOICatalog().df
m = kois.koi_disposition == "CONFIRMED"
x, y = kois[m].koi_period, kois[m].koi_prad

x0 = np.exp(np.linspace(np.log(0.1), np.log(9e4), 500))
for f in np.exp(np.linspace(np.log(0.001), np.log(100), 8)):
    y0 = np.sqrt(f * np.sqrt(x0))
    ax.plot(x0, y0, "k", lw=0.5, alpha=0.5)

ax.loglog(x, y, ".", color=COLORS["DATA"], ms=4)
ax.set_xlim(0.5, 450.)
ax.set_ylim(0.3, 30)
ax.set_xlabel("orbital period [days]")
ax.set_ylabel("planet radius [$R_\oplus$]")

fmt = matplotlib.ticker.FormatStrFormatter("%.0f")
ax.xaxis.set_major_formatter(fmt)
ax.yaxis.set_major_formatter(fmt)

# Plot KOIs.
m = kois.koi_disposition == "CANDIDATE"
x, y = kois[m].koi_period, kois[m].koi_prad
ax.plot(x, y, ".",  # color=COLORS["MODEL_1"],
        ms=3, alpha=0.3, zorder=-1)

# Plot the solar system.
rad = np.array("2439.7 6051.8 6371.00 3389.5 69911 58232 25362 24622"
               .split(), dtype=float)
period = np.array("0.2408467 0.61519726 1.0000174 1.8808476 11.862615 "
                  "29.447498 84.016846 164.79132".split(), dtype=float)
ax.plot(period * 365, rad / rad[2], "o",  # color=COLORS["MODEL_2"],
        ms=4, mec="none")
ax.axvline(4.2*365 / 2., color="k", alpha=0.6)
ax.axvline(4.2*365, color="k", ls="dashed", alpha=0.6)

fig.set_tight_layout(True)
fig.savefig("full_sample_zoom.pdf", bbox_inches="tight")

ax.set_xlim(0.5, max(x0))
fig.savefig("full_sample.pdf", bbox_inches="tight")

fits = pd.read_csv("../../results/fits.csv")

pf = 365.25
rf = 0.0995 / 0.00915
xerr = np.array(fits[["period_uncert_minus", "period_uncert_plus"]]).T * pf
yerr = np.array(fits[["radius_uncert_minus", "radius_uncert_plus"]]).T * rf
x = np.array(fits.period) * pf
y = np.array(fits.radius) * rf
ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=".g", capsize=0)

fig.savefig("full_sample_plus_cands.pdf", bbox_inches="tight")
