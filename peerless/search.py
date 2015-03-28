# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["run_on_kicid"]

import os
import logging
import numpy as np
import pandas as pd

from .model import Model
from .pool import IPythonPool
from .catalogs import KICatalog
from .data import load_light_curves_for_kic


def run_on_kicid(kicid, base_dir=None, lc_params=None,
                 model_params=None, fit_params=None, cand_params=None):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [KIC {0}]: %(message)s".format(kicid),
    )

    # Set the default parameters.
    lc_params = dict() if lc_params is None else lc_params
    model_params = dict() if model_params is None else model_params
    fit_params = dict() if fit_params is None else fit_params
    cand_params = dict() if cand_params is None else cand_params

    # Find the stellar parameters.
    logging.info("Loading stellar parameters")
    df = KICatalog().df
    stars = df[df.kepid == kicid]
    if not len(stars):
        raise RuntimeError("Invalid KIC ID {0}".format(kicid))
    star = stars.iloc[0]

    # Download the light curves.
    logging.info("Loading light curves")
    lcs = load_light_curves_for_kic(kicid, **lc_params)
    logging.info("Found {0} light curve sections".format(len(lcs)))
    if not len(lcs):
        return None

    # Train the model.
    mass, rad = float(star.mass), float(star.radius)
    mass = mass if np.isfinite(mass) else 1.0
    rad = rad if np.isfinite(rad) else 1.0
    logging.info("mass={0} and rad={1}".format(mass, rad))
    logging.info("Training model")
    mod = Model(lcs, smass=mass, srad=rad, **model_params)
    mod.fit_all(**fit_params)

    # Find the candidates.
    candidates = mod.find_candidates(**cand_params)
    logging.info("Found {0} candidates".format(len(candidates)))

    # Save the results.
    if base_dir is not None:
        bp = os.path.join(base_dir, "{0}".format(kicid))
        try:
            os.makedirs(bp)
        except os.error:
            pass
        fn = os.path.join(bp, "model.h5")
        logging.info("Saving model results to {0}".format(fn))
        mod.to_hdf(fn)

        fn = os.path.join(bp, "candidates.csv")
        logging.info("Saving candidate list to {0}".format(fn))
        pd.DataFrame.from_records(candidates).to_csv(fn, index=False)

    return mod


def run_on_kicids(kicids, base_dir=".", poolargs=None, **kwargs):
    base_dir = os.path.abspath(base_dir)
    arglist = kicids
    dirlist = [os.path.join(base_dir, "{0}".format(i)) for i in kicids]
    poolargs = dict() if poolargs is None else poolargs
    pool = IPythonPool(**poolargs)
    pool.client[:].push(dict(
        run_on_kicid=run_on_kicid,
    ))
    pool.run(run_on_kicid, arglist, dirlist, base_dir=base_dir, **kwargs)
