#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from multiprocessing import Pool

from peerless._search import search
from peerless.catalogs import KICatalog
from peerless.data import load_light_curves_for_kic, running_median_trend


def get_peaks(kicid, tau=0.6, delete=True):
    lcs = load_light_curves_for_kic(kicid, delete=delete)

    time = []
    depth = []
    depth_ivar = []

    for lc in lcs:
        time.append(np.arange(lc.time.min(), lc.time.max(), 0.25 * tau))
        d, ivar = search(tau, time[-1], lc.time, lc.flux - 1.0, 1.0/lc.ferr**2)
        depth.append(d)
        depth_ivar.append(ivar)

    time = np.concatenate(time)
    depth = np.concatenate(depth)
    depth_ivar = np.concatenate(depth_ivar)

    s2n = depth * np.sqrt(depth_ivar)
    m = depth_ivar > 0.0
    noise = np.nan + np.zeros_like(s2n)
    noise[m] = running_median_trend(time[m], np.abs(s2n[m]), 15.0)

    m = s2n > 20 * noise
    s2n_thresh = s2n[m]
    t0_thresh = time[m]
    peaks = []

    while len(s2n_thresh):
        i = np.argmax(s2n_thresh)
        t0 = t0_thresh[i]
        peaks.append((t0, s2n_thresh[i]))
        m = np.abs(t0_thresh - t0) > 2*tau
        s2n_thresh = s2n_thresh[m]
        t0_thresh = t0_thresh[m]

    return kicid, peaks

if __name__ == "__main__":
    stlr = KICatalog().df

    # Select G and K dwarfs.
    m = (4200 <= stlr.teff) & (stlr.teff <= 6100)
    m &= stlr.radius <= 1.15

    # Only include stars with sufficient data coverage.
    m &= stlr.dataspan > 365.25*2.
    m &= stlr.dutycycle > 0.6
    m &= stlr.rrmscdpp07p5 <= 1000.
    m &= stlr.kepmag < 15.

    kepids = np.array(stlr[m].kepid)[:100]
    # kepids = [
    #     2158850,
    #     3558849,
    #     5010054,
    #     5536555,
    #     5951458,
    #     8410697,
    #     8510748,
    #     8540376,
    #     9704149,
    #     9838291,
    #     10024862,
    #     10403228,
    #     10842718,
    #     10960865,
    #     11558724,
    #     12066509,
    # ]

    open("results.txt", "w").close()
    pool = Pool()
    for i, (kepid, peaks) in enumerate(pool.imap_unordered(get_peaks, kepids)):
        if (i + 1) % 500 == 0:
            print(100 * (i + 1) / len(kepids))
        if not len(peaks):
            continue
        with open("results.txt", "a") as f:
            f.write("\n".join("{0}, {1}, {2}".format(kepid, p[0], p[1])
                              for p in peaks)
                    + "\n")

    # print(get_peaks(8410697))
