# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["load_light_curves_for_kic", "load_light_curves", "LightCurve",
           "running_median_trend"]

import os
import logging
import requests
import numpy as np
from io import BytesIO
from astropy.io import fits
from zipfile import ZipFile
from scipy.ndimage.measurements import label

from .catalogs import KOICatalog
from .settings import TEXP, PEERLESS_DATA_DIR


def load_light_curves_for_kic(kicid, remove_kois=True, **kwargs):
    if remove_kois:
        kwargs["remove_kois"] = kicid
    fn = os.path.join(PEERLESS_DATA_DIR, "data", "{0}.zip".format(kicid))
    return load_light_curves(fn, **kwargs)


def load_light_curves(fn, pdc=True, delete=False, remove_kois=False,
                      detrend_hw=2.0, inject_system=None):
    if not os.path.exists(fn):
        raise ValueError("'{0}' doesn't exist".format(fn))

    # Find any KOIs.
    if remove_kois:
        df = KOICatalog().df
        kois = df[df.kepid == remove_kois]

    # Load the light curves.
    lcs = []
    n_inj_cad = 0
    with ZipFile(fn, "r") as zf:
        for n in zf.namelist():
            with zf.open(n, "r") as f:
                content = BytesIO(f.read())
            with fits.open(content) as fts:
                # Load the meta data.
                hdr = fts[0].header
                meta = dict((k.lower(), hdr[k]) for k in hdr)
                hdr = fts[2].header
                header = dict((k, hdr[k]) for k in hdr)
                data = fts[1].data
                hdr = fts[1].header

            # Load the data.
            # data, hdr = fitsio.read(fn, header=True)
            texp = hdr["INT_TIME"] * hdr["NUM_FRM"] / (24. * 60. * 60.)
            x = data["TIME"]
            q = data["SAP_QUALITY"]
            if pdc:
                y = data["PDCSAP_FLUX"]
                yerr = data["PDCSAP_FLUX_ERR"]
            else:
                y = data["SAP_FLUX"]
                yerr = data["SAP_FLUX_ERR"]
            if np.any(y[np.isfinite(y)] < 0.0):
                logging.warning("invalid data: flux < 0")
                continue

            # Remove any KOI points.
            if remove_kois:
                for _, koi in kois.iterrows():
                    period = float(koi.koi_period)
                    t0 = float(koi.koi_time0bk) % period
                    tau = float(koi.koi_duration) / 24.
                    m = np.abs((x-t0+0.5*period) % period-0.5*period) < 0.8 * tau
                    y[m] = np.nan

            # Remove bad quality points.
            y[q != 0] = np.nan
            m = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)

            if inject_system is not None:
                model = inject_system.get_value(
                    np.ascontiguousarray(x[m], dtype=float), texp=texp)
                n_inj_cad += np.sum(model < 1.0)
                y[m] *= model

            # Deal with big gaps.
            lt = np.isfinite(x)
            lt[m] &= np.append(np.diff(x[m]) < 0.5, True)

            # Loop over contiguous chunks and build light curves.
            labels, nlabels = label(lt)
            for i in range(1, nlabels + 1):
                m0 = m & (labels == i)
                if not np.any(m0):
                    continue
                lcs.append(LightCurve(x[m0], y[m0], yerr[m0], meta, header,
                                    data["MOM_CENTR1"][m0],
                                    data["MOM_CENTR2"][m0],
                                    data["POS_CORR1"][m0],
                                    data["POS_CORR2"][m0],
                                    texp=texp,
                                    hw=detrend_hw))

            if delete and os.path.exists(fn):
                os.remove(fn)
        return lcs, n_inj_cad


def running_median_trend(x, y, hw=2.0):
    r = np.empty(len(y))
    for i, t in enumerate(x):
        inds = np.abs(x-t) < hw
        r[i] = np.median(y[inds])
    return r


class LightCurve(object):

    def __init__(self, time, flux, ferr, meta, header,
                 mom_cen_1, mom_cen_2, pos_corr_1, pos_corr_2,
                 texp=TEXP, hw=2.0):
        self.trend = running_median_trend(time, flux, hw=hw)
        self.median = np.median(flux)

        self.raw_time = np.ascontiguousarray(time, dtype=float)
        self.raw_flux = np.ascontiguousarray(flux/self.median, dtype=float)
        self.raw_ferr = np.ascontiguousarray(ferr/self.median, dtype=float)

        self.time = np.ascontiguousarray(time, dtype=float)
        self.flux = np.ascontiguousarray(flux/self.trend, dtype=float)
        self.ferr = np.ascontiguousarray(ferr/self.trend, dtype=float)

        self.mom_cen_1 = np.ascontiguousarray(mom_cen_1, dtype=float)
        self.mom_cen_2 = np.ascontiguousarray(mom_cen_2, dtype=float)
        self.pos_corr_1 = np.ascontiguousarray(pos_corr_1, dtype=float)
        self.pos_corr_2 = np.ascontiguousarray(pos_corr_2, dtype=float)

        self.meta = meta
        self.header = header
        self.footprint = self.time.max() - self.time.min()
        self.texp = texp

    def __len__(self):
        return len(self.time)
