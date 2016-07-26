# -*- coding: utf-8 -*-

from peerless.plot_setup import COLORS

import os
import sys
import h5py
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

from peerless.catalogs import KOICatalog, TargetCatalog

np.random.seed(42)
all_kois = False
if "--all-kois" in sys.argv:
    sys.argv.pop(sys.argv.index("--all-kois"))
    all_kois = True
fits_path = sys.argv[1]

fig, ax = pl.subplots(1, 1, figsize=(6, 4))

targets = TargetCatalog().df
if all_kois:
    kois = KOICatalog().df
else:
    kois = pd.merge(KOICatalog().df, targets, how="inner", on="kepid")
m = kois.koi_disposition == "CONFIRMED"
x, y = kois[m].koi_period, kois[m].koi_prad

x0 = np.exp(np.linspace(np.log(0.1), np.log(9e4), 500))
for f in np.exp(np.linspace(np.log(0.001), np.log(100), 8)):
    y0 = np.sqrt(f * np.sqrt(x0))
    ax.plot(x0, y0, "k", lw=0.5, alpha=0.5, zorder=-100)

ax.loglog(x, y, ".", color=COLORS["DATA"], ms=3)
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
ax.plot(x, y, ".", ms=3, alpha=0.3, mec="none", zorder=-1)

fig.set_tight_layout(True)
fig.savefig("full_sample_no_ss_zoom.pdf", bbox_inches="tight")

# Plot the solar system.
rad = np.array("2439.7 6051.8 6371.00 3389.5 69911 58232 25362 24622"
               .split(), dtype=float)
period = np.array("0.2408467 0.61519726 1.0000174 1.8808476 11.862615 "
                  "29.447498 84.016846 164.79132".split(), dtype=float)
ax.plot(period * 365, rad / rad[2], "s", ms=4, mec="none")
ax.axvline(4.2*365 / 2., color="k", lw=0.5)
if all_kois:
    ax.axvline(4.2*365, color="k", ls="dashed", lw=0.5)

fig.savefig("full_sample_zoom.pdf", bbox_inches="tight")

ax.set_xlim(0.5, max(x0))
fig.savefig("full_sample.pdf", bbox_inches="tight")

fits = pd.read_csv("../../results/fits.csv")

pf = 365.25
rf = 0.0995 / 0.00915
xerr = np.array(fits[["period_uncert_minus", "period_uncert_plus"]]).T * pf
xerr_2 = np.vstack((np.array(fits.period - fits.min_period) * pf,
                    np.zeros(len(fits))))
xerr_2[:, np.array(fits.n_transits > 1, dtype=bool)] = 0.0
yerr = np.array(fits[["radius_uncert_minus", "radius_uncert_plus"]]).T * rf
x = np.array(fits.period) * pf
y = np.array(fits.radius) * rf
ax.errorbar(x, y, xerr=xerr_2, fmt=",k", capsize=2, lw=0.5, zorder=-100)
ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=".g", capsize=0, lw=1.0)

fig.savefig("full_sample_plus_cands.pdf", bbox_inches="tight")
fig.savefig("full_sample_plus_cands.png", bbox_inches="tight", dpi=300)

for kicid in np.array(fits.kicid):
    fn = os.path.join(fits_path, "{0}.h5".format(kicid))
    with h5py.File(fn, "r") as f:
        nsamp, nwalk = f["chain"].shape
        s_inds = np.random.randint(nsamp, size=500)
        w_inds = np.random.randint(nwalk, size=500)
        periods = f["params"]["period"][s_inds, w_inds]
        radii = np.exp(f["chain"]["bodies[0]:ln_radius"][s_inds, w_inds])
        radii /= 0.009171
    ax.plot(periods, radii, ".g", alpha=0.1, ms=3, rasterized=True, mec="none")

fig.savefig("full_sample_plus_cands_samps.pdf", bbox_inches="tight", dpi=300)
fig.savefig("full_sample_plus_cands_samps.png", bbox_inches="tight", dpi=300)
