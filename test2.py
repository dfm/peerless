#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import peerless
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from multiprocessing import Pool

kois = pd.read_hdf("data/kois.h5", "cumulative")
m = kois.koi_period > 500.0
m &= kois.koi_depth > 300.0
m &= ((kois.koi_disposition == "CANDIDATE")
      | (kois.koi_disposition == "CONFIRMED"))
targets = kois[m]

def plot_koi(koi):
    kicid = koi.kepid
    fn = "results/{0}.h5".format(kicid)
    if not os.path.exists(fn):
        return
    print(kicid)
    lcs = peerless.load_light_curves_for_kic(kicid)
    mod = peerless.Model.from_hdf(fn, lcs)

    fig = pl.figure()
    for i, res in enumerate(mod.models):
        for r, v in zip(res["test"][::-1], res["validation"]):
            p = r["prediction"]
            y = p["predict_prob"]
            pl.plot(p["time"], y, ".", color="rgb"[i % 3], alpha=0.3)
            pl.gca().axhline(v["threshold"], color="rgb"[i % 3])

    period = float(koi.koi_period)
    t0 = float(koi.koi_time0bk) % period
    t = t0
    while t < lcs[-1].time.max():
        pl.gca().axvline(t, color="k", alpha=0.3)
        t += period

    pl.ylim(0, 1.05)
    pl.savefig("results/{0}.png".format(kicid))
    pl.close(fig)

pool = Pool()
koi_list = [k for _, k in targets.iterrows()]
pool.map(plot_koi, koi_list)
