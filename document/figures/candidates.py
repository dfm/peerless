# -*- coding: utf-8 -*-

from __future__ import division, print_function

from plot_setup import COLORS
from matplotlib import rcParams
rcParams["font.size"] = 16

import peerless
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

# Get the stellar data.
kic = peerless.catalogs.KICatalog().df

# Load the list of candidates.
cands = pd.read_csv("3k_candidates.csv")
print("{0} initial candidates".format(len(cands)))

# Remove any light curves with more than one candidate.
grouped = cands.groupby("kicid").time.count()
grouped = grouped[grouped > 1].index
cands = cands[~cands.kicid.isin(grouped)]
print("{0} candidates with dupes removed".format(len(cands)))

# Cross match against the rotation and EB lists.
rot = pd.read_csv("rot.csv")
ebs = pd.read_csv("ebs.csv", skiprows=7)
joined = pd.merge(
    pd.merge(
        pd.merge(cands, rot, left_on="kicid", right_on="KID", how="left"),
        ebs, left_on="kicid", right_on="#KIC", how="left",
        suffixes=["", "_eb"]
    ), kic, left_on="kicid", right_on="kepid", how="left",
    suffixes=["", "_kic"]
)

# Sort by nearest neighbor S/N.
joined["rank"] = (joined.nn_rp/joined.radius)**2/(joined.rrmscdpp15p0*1e-6)

# Remove the known signals.
final = joined[(~(np.isfinite(joined.Prot) | np.isfinite(joined.period)))]
final = final.sort("rank", ascending=False)
print("{0} candidates with known signals removed".format(len(final)))

# Make the plot.
fig, axes = pl.subplots(6, 2, figsize=(3, 7), sharex=True)
fig.set_tight_layout(False)

dt = 3.5
for ax, (_, row) in zip(axes.flat, final.iterrows()):
    kicid = int(row.kicid)
    t0 = float(row.time)
    lcs = peerless.data.load_light_curves_for_kic(kicid, min_break=10)
    t = np.concatenate([lc.time for lc in lcs])
    f = np.concatenate([lc.flux for lc in lcs])

    t0 = float(row.time)
    ax.axvline(0.0, color="k", alpha=0.5, lw=0.5)
    m = np.abs(t - t0) < dt
    ax.plot(t[m] - t0, f[m], ".", color=COLORS["DATA"], ms=1.5)
    ax.annotate("{0:.0f}".format(row.kicid),
                xy=(0, 0), xycoords="axes fraction",
                ha="left", va="bottom", fontsize=8,
                xytext=(3, 3), textcoords="offset points")
    ax.set_xlim(-dt, dt)
    ax.set_yticklabels([])
    ax.xaxis.set_major_locator(pl.MaxNLocator(4, prune="lower"))
    ax.yaxis.set_major_locator(pl.NullLocator())

# Make another axes object for the shared label.
big_ax = fig.add_subplot(111, axes="off")
big_ax.xaxis.set_major_locator(pl.NullLocator())
big_ax.yaxis.set_major_locator(pl.NullLocator())
big_ax.set_xlabel("time since transit [days]")
big_ax.xaxis.set_label_coords(0.5, -0.05)
big_ax.set_frame_on(False)

fig.subplots_adjust(hspace=0.0, wspace=0.0, top=0.99, right=0.98, left=0.02)
fig.savefig("candidates.pdf")
