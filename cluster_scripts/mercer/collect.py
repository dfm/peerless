#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = []

import os
import h5py
import glob
import numpy as np
import pandas as pd

all_cands = None
all_comp = []
for l, fn in enumerate(glob.iglob("*/candidates.csv")):
    # if l > 10:
    #     break

    bn = fn.split("/")[0]
    kicid = int(bn)

    # Load the completeness information.
    fn2 = os.path.join(bn, "model.h5")
    if not os.path.exists(fn2):
        continue
    try:
        with h5py.File(fn2, "r") as f:
            comp = []
            for k in f:
                g = f[k]
                ids = range(3)
                del ids[g.attrs["split_id"]]
                for i, j in zip(ids, ids[::-1]):
                    ts = g["validation_{0}".format(i)]
                    vs = g["validation_{0}".format(j)]
                    th = vs.attrs["threshold"]
                    prc = ts["precision_recall_curve"][...]
                    rec = prc["recall"][prc["threshold"] >= th]
                    if len(rec):
                        comp.append(rec[0])
                    else:
                        comp.append(0)
        all_comp.append((kicid, np.mean(comp)))
    except:
        print("Skipping {0}".format(kicid))
        continue

    # Load the candidates.
    df = pd.read_csv(fn)
    df["kicid"] = [kicid] * len(df)
    if all_cands is None:
        all_cands = df
    else:
        all_cands = all_cands.append(df, ignore_index=True)

    print(len(all_cands))

all_cands.to_csv("all_candidates.csv", index=False)

# Format the completeness data.
all_comp = np.array(all_comp, dtype=[("kicid", int), ("recall", float)])
all_comp = pd.DataFrame.from_records(all_comp)
all_comp.to_csv("all_recall.csv", index=False)
