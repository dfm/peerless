# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["load_light_curves_for_kic", "load_light_curves", "LightCurve",
           "running_median_trend"]

import os
import fitsio
import logging
import requests
import numpy as np
from scipy.ndimage.measurements import label

from .catalogs import KOICatalog
from .settings import TEXP, PEERLESS_DATA_DIR


def load_light_curves_for_kic(kicid, clobber=False, remove_kois=True,
                              **kwargs):
    # Make sure that that data directory exists.
    bp = os.path.join(PEERLESS_DATA_DIR, "data")
    try:
        os.makedirs(bp)
    except os.error:
        pass

    # Get the list of data URLs.
    urls = _get_mast_light_curve_urls(kicid)

    # Loop over the URLs and download the files if needed.
    fns = []
    for url in urls:
        fn = os.path.join(bp, url.split("/")[-1])
        fns.append(fn)
        if os.path.exists(fn) and not clobber:
            continue

        # Download the file.
        r = requests.get(url)
        if r.status_code != requests.codes.ok:
            r.raise_for_status()
        with open(fn, "wb") as f:
            f.write(r.content)

    # Load the light curves.
    if remove_kois:
        kwargs["remove_kois"] = kicid
    return load_light_curves(fns, **kwargs)


def load_light_curves(fns, pdc=True, delete=False, remove_kois=False,
                      detrend_hw=2.0):
    # Find any KOIs.
    if remove_kois:
        df = KOICatalog().df
        kois = df[df.kepid == remove_kois]
        if len(kois):
            logging.info("Removing {0} known KOIs".format(len(kois)))

    # Load the light curves.
    lcs = []
    for fn in fns:
        # Load the data.
        data, hdr = fitsio.read(fn, header=True)
        texp = hdr["INT_TIME"] * hdr["NUM_FRM"] / (24. * 60. * 60.)
        x = data["TIME"]
        q = data["SAP_QUALITY"]
        if pdc:
            y = data["PDCSAP_FLUX"]
            yerr = data["PDCSAP_FLUX_ERR"]
        else:
            y = data["SAP_FLUX"]
            yerr = data["SAP_FLUX_ERR"]

        # Load the meta data.
        hdr = fitsio.read_header(fn, 0)
        meta = dict(
            channel=hdr["CHANNEL"],
            skygroup=hdr["SKYGROUP"],
            module=hdr["MODULE"],
            output=hdr["OUTPUT"],
            quarter=hdr["QUARTER"],
            season=hdr["SEASON"],
        )

        # Remove any KOI points.
        if remove_kois:
            for _, koi in kois.iterrows():
                period = float(koi.koi_period)
                t0 = float(koi.koi_time0bk) % period
                tau = float(koi.koi_duration) / 24.
                m = np.abs((x-t0+0.5*period) % period-0.5*period) < tau
                y[m] = np.nan

        # Remove bad quality points.
        y[q != 0] = np.nan
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)

        # Loop over contiguous chunks and build light curves.
        labels, nlabels = label(np.isfinite(x))
        for i in range(1, nlabels + 1):
            m0 = m & (labels == i)
            if not np.any(m0):
                continue
            lcs.append(LightCurve(x[m0], y[m0], yerr[m0], meta, texp=texp,
                                  hw=detrend_hw))

        if delete:
            os.remove(fn)
    return lcs


def running_median_trend(x, y, hw=2.0):
    r = np.empty(len(y))
    for i, t in enumerate(x):
        inds = np.abs(x-t) < hw
        r[i] = np.median(y[inds])
    return r


class LightCurve(object):

    def __init__(self, time, flux, ferr, meta, texp=TEXP, hw=2.0):
        self.trend = running_median_trend(time, flux, hw=hw)
        self.median = np.median(flux)

        self.raw_time = np.ascontiguousarray(time, dtype=float)
        self.raw_flux = np.ascontiguousarray(flux/self.median, dtype=float)
        self.raw_ferr = np.ascontiguousarray(ferr/self.median, dtype=float)

        self.time = np.ascontiguousarray(time, dtype=float)
        self.flux = np.ascontiguousarray(flux/self.trend, dtype=float)
        self.ferr = np.ascontiguousarray(ferr/self.trend, dtype=float)

        self.meta = meta
        self.footprint = self.time.max() - self.time.min()
        self.texp = texp

    def __len__(self):
        return len(self.time)


def _get_mast_light_curve_urls(kic, short_cadence=False, **params):
    # Build the URL and request parameters.
    url = "http://archive.stsci.edu/kepler/data_search/search.php"
    params["action"] = params.get("action", "Search")
    params["outputformat"] = "JSON"
    params["coordformat"] = "dec"
    params["verb"] = 3
    params["ktc_kepler_id"] = kic
    params["ordercolumn1"] = "sci_data_quarter"
    if not short_cadence:
        params["ktc_target_type"] = "LC"

    # Get the list of files.
    r = requests.get(url, params=params)
    if r.status_code != requests.codes.ok:
        r.raise_for_status()

    # Format the data URLs.
    kic = "{0:09d}".format(kic)
    base_url = ("http://archive.stsci.edu/pub/kepler/lightcurves/{0}/{1}/"
                .format(kic[:4], kic))
    for row in r.json():
        ds = row["Dataset Name"].lower()
        tt = row["Target Type"].lower()
        yield base_url + "{0}_{1}lc.fits".format(ds, tt[0])
