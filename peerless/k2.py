# -*- coding: utf-8 -*-

from __future__ import division, print_function

import requests
import numpy as np
from io import BytesIO
from astropy.io import fits

from .data import LightCurve

__all__ = ["load_k2_light_curve"]


def load_k2_light_curve(star, tau=0.6, sigma_clip=5.0):
    star_id = "{0}".format(star.epic_number)
    url = ("https://archive.stsci.edu/missions/hlsp/everest/c{0:02d}/{1}/{2}/"
           .format(star.k2_campaign, star_id[:4] + "00000", star_id[4:]))
    url += "hlsp_everest_k2_llc_{1}-c{0:02d}_kepler_v1.0_lc.fits".format(
        star.k2_campaign, star_id)
    r = requests.get(url)
    r.raise_for_status()

    with fits.open(BytesIO(r.content)) as fts:
        hdr = fts[0].header
        meta = dict((k.lower(), hdr[k]) for k in hdr)
        hdr = fts[2].header
        header = dict((k, hdr[k]) for k in hdr)
        data = fts[1].data
        hdr = fts[1].header

    meta["skygroup"] = 0
    meta["season"] = 0
    meta["quarter"] = meta["campaign"]

    x = data["TIME"]
    y = data["FLUX"]

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    # Sigma clipping
    m = np.zeros_like(x, dtype=bool)
    diff = np.abs(y[1:-1] - 0.5 * (y[:-2] + y[2:]))
    sig = np.sqrt(np.mean(diff**2))
    m[1:-1] = diff < sigma_clip * sig
    tdiff = np.diff(x[~m])
    tdiff = np.min(np.vstack((np.append(tdiff, np.inf),
                              np.append(np.inf, tdiff))), axis=0)
    m[~m] |= tdiff < tau
    x = x[m]
    y = y[m]

    yerr = 1.4 * np.median(np.abs(np.diff(y))) + np.zeros_like(y)
    corr = np.random.randn(len(y))

    return LightCurve(x, y, yerr, meta, header, corr, corr, corr, corr)
