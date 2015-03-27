#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import time
import peerless
import numpy as np
import pandas as pd
from multiprocessing import Pool

kois = pd.read_hdf("data/kois.h5", "cumulative")
m = kois.koi_period > 1000.0
m &= kois.koi_depth > 300.0
m &= ((kois.koi_disposition == "CANDIDATE")
      | (kois.koi_disposition == "CONFIRMED"))
targets = kois[m][["kepid", "koi_smass", "koi_srad"]]
targets = [t for _, t in targets.iterrows()]


def fit_target(row):
    kicid = int(row.kepid)
    fn = "results/{0}.h5".format(kicid)

    print("Starting {0}".format(kicid))
    lcs = peerless.load_light_curves_for_kic(kicid)

    strt = time.time()
    mod = peerless.Model(lcs, smass=float(row.koi_smass),
                         srad=float(row.koi_srad), npos=20000)
    mod.fit_all(n_jobs=-1)
    mod.to_hdf(fn)
    print("Finished {0} in {1} seconds".format(kicid, time.time() - strt))


pool = Pool()
pool.map(fit_target, targets)
