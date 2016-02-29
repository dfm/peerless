# -*- coding: utf-8 -*-

from __future__ import division, print_function

from plot_setup import COLORS

import os
import pickle
import numpy as np
import matplotlib.pyplot as pl

from peerless.search import search

models = [
    ("variability", "gp", 7220674, 0),
    ("step", "step", 8631697, 0),
    ("box", "box2", 5521451, 0),
    ("transit", "transit", 8505215, 0),
]

fig, axes = pl.subplots(2, 2, figsize=(8, 8))

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
    rng = t0 + np.array([-2, 2])
    gp, y0 = peak["gps"][name]

    x, y = peak["data"][:2]
    m = (rng[0] < x) & (x < rng[1])
    x0 = np.linspace(rng[0], rng[1], 500)
    ax.plot(24*(x[m] - t0), y[m], ".", color=COLORS["DATA"])

    mu = gp.predict(y0, x0, return_cov=False)
    ax.plot(24*(x0 - t0), mu, color=COLORS["MODEL_2"], lw=2.5, alpha=0.8)
    ax.annotate("({0})".format(a), xy=(0, 0), xycoords="axes fraction",
                xytext=(5, 12), textcoords="offset points")

    ax.set_xlim(24*(rng - t0))
    ax.set_yticklabels([])
    ax.set_xlabel("hours since event")
    ax.set_title("KIC {0}".format(kicid, disp))

fig.savefig("model_comp.pdf", bbox_inches="tight")
