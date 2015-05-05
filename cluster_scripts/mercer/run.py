#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import argparse
import peerless
import numpy as np
import pandas as pd

# Parse the command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("out_dir", help="the directory for the output")
parser.add_argument("-n", "--number", default=100, type=int,
                    help="the number of targets to search")
parser.add_argument("-p", "--profile-dir", default=None,
                    help="the IPython profile dir")
parser.add_argument("--kois", action="store_true",
                    help="run on the KOIs")
args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

poolargs = dict()
if args.profile_dir is not None:
    poolargs["profile_dir"] = args.profile_dir

# Only select G-stars.
stlr = peerless.catalogs.KICatalog().df

if args.kois:
    kois = peerless.catalogs.KOICatalog().df
    kois = kois[kois.koi_pdisposition == "CANDIDATE"]
    stlr = pd.merge(kois, stlr, on="kepid", how="left")

# Shuffle the order.
stlr["order_by"] = np.random.rand(len(stlr))

select = (stlr.teff > 4100) & (stlr.teff < 6100)
select &= (stlr.logg > 4.0) & (stlr.logg < 4.9)
select &= (stlr.kepmag > 10.0) & (stlr.kepmag < 15)
kicids = np.array(stlr[select].sort("order_by").kepid)

# Set up the lock files.
fns = map(os.path.join(args.out_dir, "{0}.lock").format, kicids)
targets = [(kicid, fn) for kicid, fn in zip(kicids, fns)
           if not (os.path.exists(fn) or
                   os.path.exists(os.path.join(
                       args.out_dir, "{0}".format(kicid), "model.h5"
                   )))][:args.number]

# Create the lock files.
for _, fn in targets:
    open(fn, "w").write("locked")

# Run the search.
try:
    peerless.search.run_on_kicids([k for k, _ in targets], base_dir=args.out_dir,
                                  quiet=True, poolargs=poolargs,
                                  lc_params=dict(clobber=True))

finally:
    for _, fn in targets:
        os.remove(fn)
