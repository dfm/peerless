#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import emcee
import logging
import traceback
import numpy as np
import pandas as pd
from scipy.stats import beta
from functools import partial
import matplotlib.pyplot as pl
from multiprocessing import Pool

import george
from george import kernels, ModelingMixin
from george.modeling import check_gradient

import transit

from peerless.data import load_light_curves_for_kic
from peerless.catalogs import KICatalog, EBCatalog, KOICatalog


# Newton's constant in $R_\odot^3 M_\odot^{-1} {days}^{-2}$.
_G = 2945.4625385377644


def fit_light_curve(args, remove_kois=False, output_dir="fits", plot_all=False,
                    no_plots=False, verbose=False, quiet=False, delete=False):
    print(args)
    kicid = args["kicid"]

    # Initialize the system.
    system = transit.System(transit.Central(
        flux=1.0, radius=args["srad"], mass=args["smass"], q1=0.5, q2=0.5,
    ))
    system.add_body(transit.Body(
        radius=args["radius"],
        period=args["period"],
        t0=args["t0"],
        b=args["impact"],
        e=0.01,
        omega=0.0,
    ))
    system.thaw_parameter("*")
    system.freeze_parameter("bodies*ln_mass")

    # Load the light curves.
    lcs, _ = load_light_curves_for_kic(kicid, delete=delete,
                                       remove_kois=remove_kois)

    # Which light curves should be fit?
    fit_lcs = []
    other_lcs = []
    gps = []
    for lc in lcs:
        f = system.light_curve(lc.time, lc.texp)
        if np.any(f < 1.0):
            fit_lcs.append(lc)
            kernel = np.var(lc.flux) * kernels.Matern32Kernel(2**2)
            gp = george.GP(kernel, white_noise=2*np.log(np.mean(lc.ferr)),
                           fit_white_noise=True)
            gp.compute(lc.time, lc.ferr)
            gps.append(gp)
        else:
            other_lcs.append(lc)

    eb = beta(1.12, 3.09)

    # Probabilistic model:
    def lnprior():
        star = system.central
        return -0.5 * (
            ((star.mass - args["smass"]) / args["smass_err"]) ** 2 +
            ((star.radius - args["srad"]) / args["srad_err"]) ** 2
        ) + eb.logpdf(system.bodies[0].e)

    def lnlike():
        ll = 0.0
        for gp, lc in zip(gps, fit_lcs):
            r = lc.flux - system.light_curve(lc.time, texp=lc.texp)
            ll += gp.lnlikelihood(r, quiet=True)
            if not np.isfinite(ll):
                return -np.inf, 0

        # Compute number of cadences with transits in the other light curves.
        ncad = sum((system.light_curve(lc.time) < system.central.flux).sum()
                   for lc in other_lcs)

        return ll, ncad

    def lnprob(theta):
        system.set_vector(theta[:len(system)])
        i = len(system)
        for gp in gps:
            n = len(gp)
            gp.set_vector(theta[i:i+n])
            i += n

        lp = lnprior()
        if not np.isfinite(lp):
            return -np.inf, (0, )

        ll, ncad = lnlike()
        if not np.isfinite(ll):
            return -np.inf, (0, )

        return lp + ll, ncad

    p0 = np.concatenate([system.get_vector()]+[g.get_vector() for g in gps])
    ndim, nwalkers = len(p0), 64
    p0 = p0[None, :] + 1e-6 * np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
    sampler.run_mcmc(p0, 1000)

    return sampler


def _wrapper(*args, **kwargs):
    quiet = kwargs.pop("quiet", False)
    try:
        return fit_light_curve(*args, **kwargs)
    except:
        if not quiet:
            raise
        with open(os.path.join(kwargs.get("output_dir", "fits"),
                               "errors.txt"), "a") as f:
            f.write("{0} failed with exception:\n{1}"
                    .format(args, traceback.format_exc()))
    return [], None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="model some light curves")

    parser.add_argument("kicids", nargs="*", type=int, help="some KIC IDs")
    parser.add_argument("--candidates",
                        help="the ")
    parser.add_argument("-p", "--parallel", action="store_true",
                        help="parallelize across targets")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="log errors instead of raising")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="more output to the screen")
    parser.add_argument("-c", "--clean", action="store_true",
                        help="remove temporary light curve files")
    parser.add_argument("-o", "--output-dir", default="output",
                        help="the output directory")
    parser.add_argument("--no-plots", action="store_true",
                        help="don't make any plots")

    parser.add_argument("--no-remove-kois", action="store_true",
                        help="leave the known KOIs in the light curves")
    parser.add_argument("--fit-all", action="store_true",
                        help="fit even rejected candidates")
    parser.add_argument("--max-offset", type=float, default=10.0,
                        help="the maximum centroid offset S/N")

    args = parser.parse_args()

    # Build the dictionary of search keywords.
    function = partial(
        _wrapper,
        remove_kois=not args.no_remove_kois,
        output_dir=args.output_dir,
        no_plots=args.no_plots,
        verbose=args.verbose,
        quiet=args.quiet,
        delete=args.clean,
    )

    # Check and create the output directory.
    if os.path.exists(args.output_dir):
        logging.warning("Output directory '{0}' exists"
                        .format(args.output_dir))
    else:
        os.makedirs(args.output_dir)

    # Load the candidate list.
    cands = pd.read_csv(args.candidates)
    if len(args.kicids):
        cands = cands[cands.kicid.isin(args.kicids)]
    if not args.fit_all:
        cands = cands[cands.accept_time & cands.accept_bic]
    if not args.fit_all:
        cands = cands[cands.accept_time & cands.accept_bic]
    cands["centroid_offset_s2n"] = \
        cands.centroid_offset / cands.centroid_offset_err

    # Load the stellar catalog.
    kic = KICatalog().df
    kic = kic[kic.kepid.isin(cands.kicid)]

    # Initialize.
    inits = []
    for id_, rows in cands.groupby("kicid"):
        if not np.any(rows.centroid_offset_s2n <= args.max_offset):
            continue

        # Save the stellar parameters.
        star = kic[kic.kepid == id_]
        system = dict(
            kicid=id_,
            srad=float(star.radius),
            srad_err=0.5 * float(star.radius_err1 - star.radius_err2),
            smass=float(star.mass),
            smass_err=0.5 * float(star.mass_err1 - star.mass_err2),
        )

        # Multiple transits.
        if len(rows) > 1:
            times = np.sort(rows.transit_time)
            system["period"] = np.mean(np.diff(times))
            system["t0"] = times[0]
            row = rows.mean()
        else:
            system["period"] = 2000.0
            system["t0"] = float(rows.transit_time)
            row = rows.iloc[0]

        # Initial parameters.
        system["radius"] = float(row.transit_ror * star.radius)

        # Semi-major.
        a = float(_G*system["period"]**2*(star.mass)/(4*np.pi*np.pi)) ** (1./3)

        # Duration.
        dur = system["period"] * system["radius"] / (np.pi * a)
        b = np.sqrt(np.abs(1 - dur / float(row.transit_duration)))
        system["impact"] = b
        inits.append(system)

    # Deal with parallelization.
    if args.parallel:
        pool = Pool()
        M = pool.imap_unordered
    else:
        M = map

    # Run.
    samplers = list(M(_wrapper, inits))
