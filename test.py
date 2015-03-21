#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import time
import peerless
import numpy as np
import pandas as pd
from multiprocessing import Pool

kois = pd.read_hdf("data/kois.h5", "cumulative")
targets = np.array(kois[kois.koi_period > 500.0].sort("koi_period").kepid)


def fit_target(kicid):
    fn = "results/{0}.h5".format(kicid)

    print("Starting {0}".format(kicid))
    lcs = peerless.load_light_curves_for_kic(10287723)

    strt = time.time()
    mod = peerless.Model(lcs)
    mod.fit_all()
    mod.to_hdf(fn)
    print("Finished {0} in {1} seconds".format(kicid, time.time() - strt))


pool = Pool()
pool.map(fit_target, targets)
