# -*- coding: utf-8 -*-

from __future__ import division, print_function

from plot_setup import COLORS

import os
import pickle
import numpy as np
import matplotlib.pyplot as pl

from peerless.search import search

kicid = 8505215
# kicid = 10842718
# kicid = 10287723
# kicid = 6551440
fn = os.path.join("cache", "init-{0}.pkl".format(kicid))
if os.path.exists(fn):
    print("using cached file: {0}".format(fn))
    with open(fn, "rb") as f:
        results = pickle.load(f)
else:
    results = search((kicid, None), verbose=True, delete=False)
    with open(fn, "wb") as f:
        pickle.dump(results, f, -1)

fig, ax = pl.subplots(1, 1, figsize=(8, 4))
ax.plot(results.search_time, results.search_scalar, color=COLORS["DATA"])
ax.plot(results.search_time, results.search_background,
        color=COLORS["MODEL_2"])
ax.plot(results.search_time, results.detect_thresh*results.search_background,
        color=COLORS["MODEL_2"], ls="dashed")
ax.set_xlim(results.search_time.min(), results.search_time.max())

fig.savefig("initial_candidates.pdf", bbox_inches="tight")
