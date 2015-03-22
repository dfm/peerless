#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import peerless
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from multiprocessing import Pool

kois = pd.read_hdf("data/kois.h5", "cumulative")
targets = kois[kois.kepid == 10287723]  # [kois.koi_period > 500.0]

for _, koi in targets.iterrows():
    kicid = koi.kepid
    fn = "results/{0}.h5".format(kicid)
    lcs = peerless.load_light_curves_for_kic(kicid)
    mod = peerless.Model.from_hdf(fn, lcs)

    for i, r in enumerate(mod.models):
        pl.plot(r["results"]["time"], r["results"]["predict_prob"],
                "rb"[i % 2], alpha=0.5)
        pl.plot(r["results"]["time"],
                r["threshold"]+np.zeros(len(r["results"])), "br"[i%2])

    period = float(koi.koi_period)
    t0 = float(koi.koi_time0bk) % period
    t = t0
    while t < lcs[-1].time.max():
        pl.gca().axvline(t, color="k")
        t += period

    pl.savefig("plot.png")
    assert 0


def fit_target(kicid):
    fn = "results/{0}.h5".format(kicid)

    print("Starting {0}".format(kicid))
    lcs = peerless.load_light_curves_for_kic(kicid)

    strt = time.time()
    mod = peerless.Model(lcs)
    mod.fit_all()
    mod.to_hdf(fn)
    print("Finished {0} in {1} seconds".format(kicid, time.time() - strt))


pool = Pool()
pool.map(fit_target, targets)
