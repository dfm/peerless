# -*- coding: utf-8 -*-

from __future__ import division, print_function

from peerless.plot_setup import COLORS

import os
import pickle
import numpy as np
import matplotlib.pyplot as pl

from peerless.search import search

models = [
    ("step", "step", 8631697, 0),
    ("variability", "gp", 7220674, 0),
    ("box", "box1", 9411471, 0),
    # ("box", "box2", 5521451, 0),
    ("transit", "transit", 8505215, 0),
]

fig, axes = pl.subplots(2, 2, figsize=(8.8, 8))

os.makedirs("cache", exist_ok=True)
for a, ax, (disp, name, kicid, peak_id) in zip("abcd", axes.flatten(), models):
    fn = os.path.join("cache", "{0}.pkl".format(kicid))
    if os.path.exists(fn):
        print("using cached file: {0}".format(fn))
        with open(fn, "rb") as f:
            results = pickle.load(f)
    else:
        results = search((kicid, None), detect_thresh=15, verbose=True,
                         all_models=True)
        with open(fn, "wb") as f:
            pickle.dump(results, f, -1)

    peak = results.peaks[peak_id]
    t0 = peak["transit_time"]
    gp, y0 = peak["gps"][name]
    if disp == "step":
        t0 = gp.mean["t0"]
    elif disp == "box":
        t0 = 0.5 * (gp.mean.mn + gp.mean.mx)
    rng = t0 + np.array([-2, 2])

    x, y = peak["data"][:2]
    m = (rng[0] < x) & (x < rng[1])
    x0 = np.linspace(rng[0], rng[1], 500)
    mean = np.median(y[m])
    ax.plot(24*(x[m] - t0), (y[m] - mean)*1e3, ".", color=COLORS["DATA"])

    mu = gp.predict(y0, x0, return_cov=False)
    ax.plot(24*(x0 - t0), (mu - mean) * 1e3, color=COLORS["MODEL_2"], lw=2.5,
            alpha=0.8)
    ax.annotate("({0}) {1}".format(a, disp),
                xy=(0, 0), xycoords="axes fraction",
                xytext=(5, 12), textcoords="offset points")

    mn, mx = (np.min(y[m]) - mean) * 1e3, (np.max(y[m]) - mean) * 1e3
    delta, mid = 0.5 * 1.1 * (mx - mn), 0.5 * (mx + mn)
    ax.set_ylim(mid - delta, mid + delta)
    ax.set_xlim(24*(rng - t0))
    ax.yaxis.set_major_locator(pl.MaxNLocator(5))
    if a in "ac":
        ax.set_ylabel("relative flux [ppt]")
        ax.yaxis.set_label_coords(-0.25, 0.5)
    ax.set_xlabel("hours since event")
    ax.set_title("KIC {0}".format(kicid, disp))

fig.savefig("model_comp.pdf", bbox_inches="tight")
