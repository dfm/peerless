#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import numpy as np
from functools import partial
import matplotlib.pyplot as pl
from multiprocessing import Pool

from peerless._search import search
from peerless.catalogs import KICatalog
from peerless.data import (load_light_curves_for_kic, running_median_trend,
                           load_light_curves)


def get_peaks(kicid=None,
              lcs=None,
              tau=0.6,
              detrend_hw=2.0,
              remove_kois=True,
              grid_frac=0.25,
              noise_hw=15.0,
              detect_thresh=20.0,
              output_dir="output",
              delete=True):
    """
    :param tau:
        The transit duration. (default: 0.6)

    :param detrend_hw:
        Half width of running window for de-trending. (default: 2.0)

    :param remove_kois:
        Remove data points near known KOI transits. (default: True)

    :param grid_frac:
        The search grid spacing as a fraction of the duration. (default: 0.25)

    :param noise_hw:
        Half width of running window for noise estimation. (default: 15.0)

    :param detect_thresh:
        Relative S/N detection threshold. (default: 20.0)

    :param delete:
        Delete the light curve files after loading them. (default: True)

    """
    if lcs is None and kicid is None:
        raise ValueError("you must specify 'lcs' or 'kicid'")
    if lcs is None:
        lcs = load_light_curves_for_kic(kicid, delete=delete,
                                        remove_kois=remove_kois)
    else:
        kicid = "unknown-target"

    # Loop over light curves and search each one.
    time = []
    depth = []
    depth_ivar = []
    for lc in lcs:
        time.append(np.arange(lc.time.min(), lc.time.max(), grid_frac * tau))
        d, ivar = search(tau, time[-1], lc.time, lc.flux - 1.0, 1.0/lc.ferr**2)
        depth.append(d)
        depth_ivar.append(ivar)
    time = np.concatenate(time)
    depth = np.concatenate(depth)
    depth_ivar = np.concatenate(depth_ivar)

    # Compute the depth S/N time series and smooth it to estimate a background
    # noise level.
    s2n = depth * np.sqrt(depth_ivar)
    m = depth_ivar > 0.0
    noise = np.nan + np.zeros_like(s2n)
    noise[m] = running_median_trend(time[m], np.abs(s2n[m]), noise_hw)

    # Find peaks about the fiducial threshold.
    m = s2n > detect_thresh * noise
    s2n_thresh = s2n[m]
    t0_thresh = time[m]
    noise_thresh = noise[m]
    peaks = []
    while len(s2n_thresh):
        i = np.argmax(s2n_thresh)
        t0 = t0_thresh[i]
        peaks.append((kicid, t0 + 0.5 * tau, s2n_thresh[i], noise_thresh[i]))

        m = np.abs(t0_thresh - t0) > 2*tau
        s2n_thresh = s2n_thresh[m]
        t0_thresh = t0_thresh[m]
        noise_thresh = noise_thresh[m]

    # For each peak, plot the diagnostic plots.
    d = os.path.join(output_dir, kicid)
    os.makedirs(d, exist_ok=True)
    for i, peak in enumerate(peaks):
        t0 = peak[1]

        fig, axes = pl.subplots(3, 2, figsize=(10, 6))

        # Raw flux.
        row = axes[0]
        for ax in row:
            [ax.plot(lc.raw_time, lc.raw_flux, "-k") for lc in lcs]
            ax.set_xticklabels([])

        # De-trended flux.
        row = axes[1]
        for ax in row:
            [ax.plot(lc.time, lc.flux, "-k") for lc in lcs]
            ax.set_xticklabels([])

        # Periodogram.
        row = axes[2]
        for ax in row:
            ax.plot(time + 0.5*tau, s2n, "k")
            ax.plot(time + 0.5*tau, noise, "g")
            ax.plot(time + 0.5*tau, detect_thresh * noise, ":g")

        for ax1, ax2 in axes:
            ax1.set_xlim(time.min() - 5.0, time.max() + 5.0)
            ax1.axvline(t0, color="g", lw=5, alpha=0.3)

            ax2.set_xlim(t0 - 5.0, t0 + 5.0)
            ax2.axvline(t0, color="g", lw=5, alpha=0.3)
            ax2.axvline(t0 - 0.5*tau, color="k", ls="dashed")
            ax2.axvline(t0 + 0.5*tau, color="k", ls="dashed")

        fig.tight_layout()
        fig.savefig(os.path.join(d, "{0:04d}.png".format(i + 1)))
        pl.close(fig)

    return peaks

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="search for single transits")

    parser.add_argument("kicids", nargs="*", help="some KIC IDs")
    parser.add_argument("--max-targets", type=int,
                        help="the maximum number of targets")
    parser.add_argument("-f", "--filenames", nargs="+",
                        help="some light curve filenames")
    parser.add_argument("-p", "--parallel", action="store_true",
                        help="parallelize across targets")
    parser.add_argument("-o", "--output-dir", default="output",
                        help="the output directory")

    # Preset target lists:
    parser.add_argument("--planet-hunters", action="store_true",
                        help="search the planet hunter targets")
    parser.add_argument("--bright-g-dwarfs", action="store_true",
                        help="search bright G and K dwarfs")

    # Search parameters:
    parser.add_argument("--duration", type=float,  default=0.6,
                        help="the transit duration in days")
    parser.add_argument("--detrend-hw", type=float,  default=2.0,
                        help="the half width of the de-trending window")
    parser.add_argument("--dont-remove-kois", action="store_true",
                        help="leave the known KOIs in the light curves")
    parser.add_argument("--grid-frac", type=float,  default=0.25,
                        help="search grid spacing in units of the duration")
    parser.add_argument("--noise-hw", type=float,  default=15.0,
                        help="the half width of the noise estimation window")
    parser.add_argument("--detect-thresh", type=float,  default=20.0,
                        help="the relative detection threshold")

    args = parser.parse_args()

    # Build the dictionary of search keywords.
    function = partial(
        get_peaks,
        tau=args.duration,
        detrend_hw=args.detrend_hw,
        remove_kois=not args.dont_remove_kois,
        grid_frac=args.grid_frac,
        noise_hw=args.noise_hw,
        detect_thresh=args.detect_thresh,
        output_dir=args.output_dir,
    )

    # Build the list of KIC IDs.
    kicids = args.kicids

    # Presets.
    if args.planet_hunters:
        kicids += [
            2158850, 3558849, 5010054, 5536555, 5951458, 8410697, 8510748,
            8540376, 9704149, 9838291, 10024862, 10403228, 10842718, 10960865,
            11558724, 12066509,
        ]
    if args.bright_g_dwarfs:
        stlr = KICatalog().df
        m = (4200 <= stlr.teff) & (stlr.teff <= 6100)
        m &= stlr.radius <= 1.15
        m &= stlr.dataspan > 365.25*2.
        m &= stlr.dutycycle > 0.6
        m &= stlr.rrmscdpp07p5 <= 1000.
        m &= stlr.kepmag < 15.
        kicids += list(np.array(stlr[m].kepid))

    # Limit the target list.
    if args.max_targets is not None:
        if len(kicids) > args.max_targets:
            logging.warn("Truncating target list from {0} to {1}".format(
                len(kicids), args.max_targets
            ))
        kicids = kicids[:args.max_targets]

    # Check and create the output directory.
    if os.path.exists(args.output_dir):
        logging.warn("Output directory '{0}' exists".format(args.output_dir))
    else:
        os.makedirs(args.output_dir)
    cand_fn = os.path.join(args.output_dir, "candidates.csv")
    with open(cand_fn, "w") as f:
        f.write("# {0}\n".format(
            ", ".join(["kicid", "time", "s2n", "s2n_bkg"])
        ))

    if len(kicids):
        # Deal with parallelization.
        if args.parallel:
            pool = Pool()
            M = pool.imap_unordered
        else:
            M = map

        for i, peaks in enumerate(M(function, kicids)):
            if (i + 1) % 500 == 0:
                sys.stderr.write('\rdone {0:%}'.format((i + 1)/len(kicids)))
            if not len(peaks):
                continue
            with open(cand_fn, "a") as f:
                f.write("\n".join(", ".join(map("{0}".format, p))
                                  for p in peaks) + "\n")

    if args.filenames is not None:
        lcs = load_light_curves(
            args.filenames,
            detrend_hw=args.detrend_hw,
            remove_kois=not args.dont_remove_kois,
        )
        peaks = function(lcs=lcs)
        if len(peaks):
            with open(cand_fn, "a") as f:
                f.write("\n".join(", ".join(map("{0}".format, p))
                                  for p in peaks) + "\n")
