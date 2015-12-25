#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import traceback
import numpy as np
from scipy.stats import beta
from functools import partial
import matplotlib.pyplot as pl
from multiprocessing import Pool
from scipy.optimize import minimize
from collections import OrderedDict

from tqdm import tqdm

import george
from george import kernels, ModelingMixin
from george.modeling import check_gradient

import transit

from peerless._search import search
from peerless.catalogs import KICatalog, EBCatalog, KOICatalog
from peerless.data import (load_light_curves_for_kic, running_median_trend,
                           load_light_curves)


def get_peaks(kicid=None,
              lcs=None,
              inject=False,
              tau=0.6,
              detrend_hw=2.0,
              remove_kois=True,
              grid_frac=0.25,
              noise_hw=15.0,
              detect_thresh=25.0,
              max_fit_data=500,
              max_peaks=3,
              min_datapoints=10,
              output_dir="output",
              plot_all=False,
              no_plots=False,
              delete=True,
              verbose=False):
    """
    :param tau:
        The transit duration. (default: 0.6)

    :param detrend_hw:
        Half width of running window for de-trending. (default: 2.0)

    :param inject:
        Inject a transit into the light curves. (default: False)

    :param remove_kois:
        Remove data points near known KOI transits. (default: True)

    :param grid_frac:
        The search grid spacing as a fraction of the duration. (default: 0.25)

    :param noise_hw:
        Half width of running window for noise estimation. (default: 15.0)

    :param detect_thresh:
        Relative S/N detection threshold. (default: 25.0)

    :param max_fit_data:
        The maximum number of data points for fitting. (default: 500)

    :param max_peaks:
        The maximum number of peaks to analyze in detail. (default: 3)

    :param min_datapoints:
        The minimum number of in-transit data points. (default: 10)

    :param output_dir:
        The parent directory for the plots. (default: output)

    :param plot_all:
        Make all the plots instead of just the transit plots. (default: False)

    :param no_plots:
        Suppress all plotting. (default: False)

    :param delete:
        Delete the light curve files after loading them. (default: True)

    :param verbose:
        Moar printing. (default: False)

    """
    system = None
    if inject:
        system = transit.System(transit.Central(q1=np.random.rand(),
                                                q2=np.random.rand()))
        r = np.exp(np.random.uniform(np.log(0.04), np.log(0.2)))
        system.add_body(transit.Body(
            radius=r,
            period=np.exp(np.random.uniform(np.log(3*365), np.log(15*365))),
            b=np.random.uniform(0, 1+r),
            e=beta.rvs(0.867, 3.03),
            omega=np.random.uniform(0, 2*np.pi),
            t0=np.random.uniform(120, 1600),
        ))
        injection = dict(
            t0=system.bodies[0].t0,
            period=system.bodies[0].period,
            ror=system.bodies[0].radius,
            b=system.bodies[0].b,
            e=system.bodies[0].e,
            omega=system.bodies[0].omega,
        )

    if lcs is None and kicid is None:
        raise ValueError("you must specify 'lcs' or 'kicid'")
    if lcs is None:
        lcs = load_light_curves_for_kic(kicid, delete=delete,
                                        detrend_hw=detrend_hw,
                                        remove_kois=remove_kois,
                                        inject_system=system)
    else:
        kicid = "unknown-target"

    # Loop over light curves and search each one.
    time = []
    chunk = []
    depth = []
    depth_ivar = []
    s2n = []
    for i, lc in enumerate(lcs):
        time.append(np.arange(lc.time.min(), lc.time.max(), grid_frac * tau))
        d, ivar, s = search(tau, time[-1], lc.time, lc.flux-1, 1/lc.ferr**2)
        depth.append(d)
        depth_ivar.append(ivar)
        s2n.append(s)
        chunk.append(i + np.zeros(len(time[-1]), dtype=int))
    time = np.concatenate(time)
    chunk = np.concatenate(chunk)
    depth = np.concatenate(depth)
    depth_ivar = np.concatenate(depth_ivar)
    s2n = np.concatenate(s2n)

    # Compute the depth S/N time series and smooth it to estimate a background
    # noise level.
    m = depth_ivar > 0.0
    noise = np.nan + np.zeros_like(s2n)
    noise[m] = running_median_trend(time[m], np.abs(s2n[m]), noise_hw)

    # Find peaks above the fiducial threshold.
    m = s2n > detect_thresh * noise
    peaks = []
    while np.any(m):
        i = np.argmax(s2n[m])
        t0 = time[m][i]
        peaks.append(dict(
            kicid=kicid,
            t0=t0 + 0.5 * tau,
            s2n=s2n[m][i],
            bkg=noise[m][i],
            depth=depth[m][i],
            depth_ivar=depth_ivar[m][i],
            chunk=chunk[m][i],
        ))
        m &= np.abs(time - t0) > 2*tau

    if verbose:
        print("Found {0} raw peaks".format(len(peaks)))

    if not len(peaks):
        return []

    for i, peak in enumerate(peaks):
        peak["num_peaks"] = len(peaks)
        peak["peak_id"] = i

    if verbose and len(peaks) > max_peaks:
        logging.warning("truncating peak list")
    peaks = peaks[:max_peaks]

    # For each peak, plot the diagnostic plots and vet.
    basedir = os.path.join(output_dir, "{0}".format(kicid))
    final_peaks = []
    for i, peak in enumerate(peaks):
        # Vetting.
        t0 = peak["t0"]
        d = peak["depth"]
        chunk = peak["chunk"]
        lc0 = lcs[chunk]
        x = lc0.raw_time
        y = lc0.raw_flux
        yerr = lc0.raw_ferr
        ndata = np.sum(np.abs(x - t0) < 0.5*tau)
        if ndata < min_datapoints:
            if verbose:
                logging.warning("there are only {0} data points in transit"
                                .format(ndata))
            continue

        # Limit number of data points in chunk.
        inds = np.sort(np.argsort(np.abs(t0 - x))[:max_fit_data])
        x = np.ascontiguousarray(x[inds])
        y = np.ascontiguousarray(y[inds])
        yerr = np.ascontiguousarray(yerr[inds])

        if verbose:
            print("{0} data points in chunk".format(len(x)))

        for k in ["channel", "skygroup", "module", "output", "quarter",
                  "season"]:
            peak[k] = lc0.meta[k]

        peak["chunk_min_time"] = x.min()
        peak["chunk_max_time"] = x.max()

        # Mean models:
        # 1. constant
        constant = np.median(y)

        # 2. transit
        m = np.abs(x - t0) < tau
        ind = np.arange(len(x))[m][np.argmin(y[m])]
        system = transit.SimpleSystem(
            period=3000.0,
            t0=x[ind],
            ror=np.sqrt(max(d, 1.0 - y[ind])),
            duration=tau,
            impact=0.5,
        )
        system.freeze_parameter("ln_period")
        system.freeze_parameter("impact")
        best = (np.inf, 0.0)
        for dur in np.linspace(0.1*tau, 2*tau, 50):
            system.duration = dur
            d = np.sum((y - system.get_value(x))**2)
            if d < best[0]:
                best = (d, dur)
        system.duration = best[1]

        # 3. step
        best = (np.inf, 0, 0.0, 0.0)
        n = 2
        for ind in range(1, len(y) - 2*n):
            a = slice(ind, ind+n)
            b = slice(ind+n, ind+2*n)
            step = StepModel(
                value1=np.mean(y[:ind]),
                value2=np.mean(y[ind+2*n:]),
                height1=np.mean(y[a]) - np.mean(y[:ind]),
                height2=np.mean(y[ind+2*n:]) - np.mean(y[b]),
                log_width_plus=0.0,
                log_width_minus=0.0,
                t0=0.5*(x[ind+n-1] + x[ind+n]),
            )

            best_minus = (np.inf, 0.0)
            m = x < step.t0
            for w in np.linspace(-4, 2, 20):
                step.log_width_minus = w
                d = np.sum((y[m] - step.get_value(x[m]))**2)
                if d < best_minus[0]:
                    best_minus = (d, w)

            best_plus = (np.inf, 0.0)
            m = x >= step.t0
            for w in np.linspace(-4, 2, 20):
                step.log_width_plus = w
                d = np.sum((y[m] - step.get_value(x[m]))**2)
                if d < best_plus[0]:
                    best_plus = (d, w)

            d = best_minus[0] + best_plus[0]
            if d < best[0]:
                best = (d, ind, best_minus[1], best_plus[1])

        _, ind, wm, wp = best
        a = slice(ind, ind+n)
        b = slice(ind+n, ind+2*n)
        step = StepModel(
            value1=np.mean(y[:ind]),
            value2=np.mean(y[ind+2*n:]),
            height1=np.mean(y[a]) - np.mean(y[:ind]),
            height2=np.mean(y[ind+2*n:]) - np.mean(y[b]),
            log_width_plus=wp,
            log_width_minus=wm,
            t0=0.5*(x[ind+n-1] + x[ind+n]),
        )
        check_gradient(step, x)

        # 4. box:
        inds = np.argsort(np.diff(y))
        inds = np.sort([inds[0], inds[-1]])
        boxes = []
        for tmn, tmx in (0.5 * (x[inds] + x[inds + 1]),
                         (t0-0.5*tau, t0+0.5*tau)):
            boxes.append(BoxModel(tmn, tmx, data=(x, y)))
            check_gradient(boxes[-1], x)

        # Loop over models and compare them.
        models = OrderedDict([
            ("transit", system),
            ("box1", boxes[1]),
            ("step", step),
            ("gp", constant),
            ("box2", boxes[0]),
        ])
        preds = dict()
        for name, mean_model in models.items():
            kernel = np.var(y) * kernels.Matern32Kernel(2**2)
            gp = george.GP(kernel, mean=mean_model, fit_mean=True,
                           white_noise=2*np.log(np.mean(yerr)),
                           fit_white_noise=True)
            gp.compute(x, yerr)

            # Set up some bounds.
            bounds = gp.get_bounds()
            n = gp.get_parameter_names()

            if name == "transit":
                bounds[n.index("mean:t0")] = (t0 - 0.5*tau, t0 + 0.5*tau)
                bounds[n.index("mean:q1_param")] = (-10, 10)
                bounds[n.index("mean:q2_param")] = (-10, 10)

            bounds[n.index("kernel:k2:ln_M_0_0")] = (np.log(0.1), None)
            bounds[n.index("white:value")] = (2*np.log(0.5*np.median(yerr)),
                                              None)
            bounds[n.index("kernel:k1:ln_constant")] = \
                (2*np.log(np.median(yerr)), None)

            # Optimize.
            initial_vector = np.array(gp.get_vector())
            r = minimize(gp.nll, gp.get_vector(), jac=gp.grad_nll, args=(y,),
                         method="L-BFGS-B", bounds=bounds)
            gp.set_vector(r.x)
            preds[name] = gp.predict(y, x, return_cov=False)

            # Initialize one of the boxes using the transit shape.
            if name == "transit":
                models["box1"].mn = system.t0 - 0.5*system.duration
                models["box1"].mx = system.t0 + 0.5*system.duration

            # Compute the -0.5*BIC.
            peak["lnlike_{0}".format(name)] = -r.fun
            peak["bic_{0}".format(name)] = -r.fun-0.5*len(r.x)*np.log(len(x))

            if verbose:
                print("Peak {0}:".format(i))
                print("Model: '{0}'".format(name))
                print("Converged: {0}".format(r.success))
                if not r.success:
                    print("Message: {0}".format(r.message))
                print("Log-likelihood: {0}"
                      .format(peak["lnlike_{0}".format(name)]))
                print("-0.5 * BIC: {0}"
                      .format(peak["bic_{0}".format(name)]))
                print("Parameters:")
                for k, v0, v in zip(gp.get_parameter_names(), initial_vector,
                                    gp.get_vector()):
                    print("  {0}: {1:.4f} -> {2:.4f}".format(k, v0, v))
                print()

            if peak["bic_{0}".format(name)] > peak["bic_transit"]:
                break

            # Deal with outliers.
            if name != "gp":
                continue
            N = len(r.x) + 1
            peak["lnlike_outlier"] = -r.fun
            peak["bic_outlier"] = -r.fun-0.5*N*np.log(len(x))
            best = (-r.fun, 0)
            for j in np.arange(len(x))[np.abs(x - t0) < 0.5*tau]:
                y0 = np.array(y)
                y0[j] = np.median(y[np.arange(len(y)) != j])
                ll = gp.lnlikelihood(y0)
                if ll > best[0]:
                    best = (ll, j)

            # Optimize the outlier model:
            m = np.arange(len(y)) != best[1]
            y0 = np.array(y)
            y0[~m] = np.median(y0[m])
            kernel = np.var(y0) * kernels.Matern32Kernel(2**2)
            gp = george.GP(kernel, mean=np.median(y0), fit_mean=True,
                           white_noise=2*np.log(np.mean(yerr)),
                           fit_white_noise=True)
            gp.compute(x, yerr)

            r = minimize(gp.nll, gp.get_vector(), jac=gp.grad_nll, args=(y0,),
                         method="L-BFGS-B", bounds=bounds)
            gp.set_vector(r.x)
            peak["lnlike_outlier"] = -r.fun
            peak["bic_outlier"] = -r.fun-0.5*N*np.log(len(x))
            preds["outlier"] = gp.predict(y0, x, return_cov=False)

            if verbose:
                print("Peak {0}:".format(i))
                print("Model: 'outlier'")
                print("Converged: {0}".format(r.success))
                print("Log-likelihood: {0}".format(peak["lnlike_outlier"]))
                print("-0.5 * BIC: {0}".format(peak["bic_outlier"]))
                print("Parameters:")
                for k, v in zip(gp.get_parameter_names(), gp.get_vector()):
                    print("  {0}: {1:.4f}".format(k, v))
                print()

        # Save the transit parameters.
        peak["transit_duration"] = system.duration
        peak["transit_ror"] = system.ror
        peak["transit_time"] = system.t0

        # Save the injected parameters.
        if inject:
            for k in ["t0", "period", "ror", "b", "e", "omega"]:
                peak["injected_{0}".format(k)] = injection[k]

            # Check for recovery.
            p = injection["period"]
            d = (peak["transit_time"] - injection["t0"] + 0.5*p) % p - 0.5*p
            peak["recovered"] = np.abs(d) < peak["transit_duration"]

        # Accept the peak?
        accept_bic = all(
            peak["bic_transit"] >= peak.get("bic_{0}".format(k), -np.inf)
            for k in models
        ) and (peak["bic_transit"] > peak["bic_outlier"])
        accept_time = (
            (peak["transit_time"] - 1.0*peak["transit_duration"]
             > peak["chunk_min_time"]) and
            (peak["transit_time"] + 1.0*peak["transit_duration"]
             < peak["chunk_max_time"])
        )
        accept = accept_bic and accept_time
        peak["accept_bic"] = accept_bic
        peak["accept_time"] = accept_time
        final_peaks.append(peak)

        if (not accept) and (not plot_all):
            continue

        # Plots.
        fig, axes = pl.subplots(3, 2, figsize=(10, 8))

        # Raw flux.
        row = axes[0]
        for ax in row:
            [ax.plot(lc.raw_time, (lc.raw_flux-1)*1e3, ".k") for lc in lcs]

            ax.set_xticklabels([])
            ax.yaxis.set_major_locator(pl.MaxNLocator(4))
            ax.xaxis.set_major_locator(pl.MaxNLocator(5))

        row[0].set_ylabel("raw [ppt]")
        ax = row[1]

        if "gp" in preds:
            ax.plot(x, (preds["gp"]-1)*1e3, "g", lw=1.5)
        if "outlier" in preds:
            ax.plot(x, (preds["outlier"]-1)*1e3, "--g", lw=1.5)
        ax.plot(x, (system.get_value(x)-1)*1e3, "r", lw=1.5)
        ax.plot(x, (step.get_value(x)-1)*1e3, "b", lw=1.5)
        [ax.plot(x, (bx.get_value(x)-1)*1e3, "m", lw=1.5) for bx in boxes]

        # De-trended flux.
        row = axes[1]
        for ax in row:
            [ax.plot(lc.time, (lc.flux-1)*1e3, ".k") for lc in lcs]
            ax.set_xticklabels([])
            ax.yaxis.set_major_locator(pl.MaxNLocator(4))
            ax.xaxis.set_major_locator(pl.MaxNLocator(5))
        row[0].set_ylabel("de-trended [ppt]")

        # Periodogram.
        row = axes[2]
        for ax in row:
            ax.plot(time + 0.5*tau, s2n, "k")
            ax.plot(time + 0.5*tau, noise, "g")
            ax.plot(time + 0.5*tau, detect_thresh * noise, ":g")
            ax.yaxis.set_major_locator(pl.MaxNLocator(4))
            ax.xaxis.set_major_locator(pl.MaxNLocator(5))
            ax.set_xlabel("time [KBJD]")
        row[0].set_ylabel("s/n")

        for ax1, ax2 in axes:
            ax1.yaxis.set_label_coords(-0.15, 0.5)

            ax1.set_xlim(time.min() - 5.0, time.max() + 5.0)
            ax1.axvline(t0, color="g", lw=5, alpha=0.3)

            ax2.set_xlim(t0 - 5.0, t0 + 5.0)
            ax2.axvline(t0, color="g", lw=5, alpha=0.3)
            ax2.axvline(t0 - 0.5*tau, color="k", ls="dashed")
            ax2.axvline(t0 + 0.5*tau, color="k", ls="dashed")
            ax2.set_yticklabels([])

        fig.subplots_adjust(
            left=0.15, bottom=0.1, right=0.98, top=0.97,
            wspace=0.05, hspace=0.12
        )
        os.makedirs(basedir, exist_ok=True)
        fig.savefig(os.path.join(basedir, "{0:04d}.png".format(i + 1)))
        pl.close(fig)

    return final_peaks


def _wrapper(*args, **kwargs):
    quiet = kwargs.pop("quiet", False)
    try:
        return get_peaks(*args, **kwargs)
    except:
        if not quiet:
            raise
        with open(os.path.join(kwargs.get("output_dir", "output"),
                               "errors.txt"), "a") as f:
            f.write("{0} failed with exception:\n{1}"
                    .format(args, traceback.format_exc()))
    return []


class StepModel(ModelingMixin):

    def get_value(self, t):
        dt_plus = (t - self.t0) * np.exp(-self.log_width_plus)
        dt_minus = (t - self.t0) * np.exp(-self.log_width_minus)
        v = (self.value1 + self.height1 * np.exp(dt_minus))*(t < self.t0)
        v += (self.value2 - self.height2 * np.exp(-dt_plus))*(t >= self.t0)
        return v

    @ModelingMixin.parameter_sort
    def get_gradient(self, t):
        h1 = self.height1
        h2 = self.height2
        wp = np.exp(self.log_width_plus)
        wm = np.exp(self.log_width_minus)
        dtp = (t - self.t0) / wp
        dtm = (t - self.t0) / wm
        ep = np.exp(-dtp)
        em = np.exp(dtm)
        mm = t < self.t0
        mp = t >= self.t0

        return dict(
            value1=np.ones_like(t) * mm,
            value2=np.ones_like(t) * mp,
            height1=em*mm,
            height2=-ep*mp,
            log_width_minus=-h1*em*dtm*mm,
            log_width_plus=-h2*ep*dtp*mp,
            t0=-(h1*em*mm/wm + h2*ep*mp/wp),
        )


class BoxModel(ModelingMixin):

    def __init__(self, mn, mx, data=None, **kwargs):
        self.mn = mn
        self.mx = mx
        if data is not None:
            t, y = data
            a = t <= self.mn
            b = self.mx < t
            c = ~(a | b)
            kwargs = dict(
                before_value=np.mean(y[a]) if np.any(a) else 0.0,
                in_value=np.mean(y[c]) if np.any(c) else 0.0,
                after_value=np.mean(y[b]) if np.any(b) else 0.0,
            )
        super(BoxModel, self).__init__(**kwargs)

    def get_value(self, t):
        a = t <= self.mn
        b = self.mx < t
        c = ~(a | b)
        return self.before_value*a + self.in_value*c + self.after_value*b

    @ModelingMixin.parameter_sort
    def get_gradient(self, t):
        a = t <= self.mn
        b = self.mx < t
        c = ~(a | b)
        return dict(
            before_value=np.ones_like(t) * a,
            in_value=np.ones_like(t) * c,
            after_value=np.ones_like(t) * b,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="search for single transits")

    parser.add_argument("kicids", nargs="*", help="some KIC IDs")
    parser.add_argument("--inject", action="store_true",
                        help="inject transits into the light curves")
    parser.add_argument("--include-ebs", action="store_true",
                        help="by default known EBs are excluded")
    parser.add_argument("--max-targets", type=int,
                        help="the maximum number of targets")
    parser.add_argument("-f", "--filenames", nargs="+",
                        help="some light curve filenames")
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
    parser.add_argument("--plot-all", action="store_true",
                        help="make all the plots")
    parser.add_argument("--no-plots", action="store_true",
                        help="don't make any plots")

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
    parser.add_argument("--no-remove-kois", action="store_true",
                        help="leave the known KOIs in the light curves")
    parser.add_argument("--grid-frac", type=float,  default=0.25,
                        help="search grid spacing in units of the duration")
    parser.add_argument("--noise-hw", type=float,  default=15.0,
                        help="the half width of the noise estimation window")
    parser.add_argument("--detect-thresh", type=float,  default=25.0,
                        help="the relative detection threshold")
    parser.add_argument("--max-fit-data", type=int,  default=500,
                        help="maximum number of points to fit")
    parser.add_argument("--max-peaks", type=int,  default=3,
                        help="the maximum number of peaks to consider")

    args = parser.parse_args()

    # Build the dictionary of search keywords.
    function = partial(
        _wrapper,
        inject=args.inject,
        tau=args.duration,
        detrend_hw=args.detrend_hw,
        remove_kois=not args.no_remove_kois,
        grid_frac=args.grid_frac,
        noise_hw=args.noise_hw,
        detect_thresh=args.detect_thresh,
        output_dir=args.output_dir,
        plot_all=args.plot_all,
        no_plots=args.no_plots,
        max_fit_data=args.max_fit_data,
        max_peaks=args.max_peaks,
        verbose=args.verbose,
        quiet=args.quiet,
        delete=args.clean,
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
    kicids = np.array(kicids, dtype=int)

    # Limit the target list.
    if args.max_targets is not None:
        if len(kicids) > args.max_targets:
            logging.warning("Truncating target list from {0} to {1}".format(
                len(kicids), args.max_targets
            ))
        kicids = kicids[:args.max_targets]

    # Remove EBs from the target list.
    if not args.include_ebs:
        # Start with EB catalog:
        ebs = set(np.array(EBCatalog().df["#KIC"]))
        l0 = len(kicids)
        kicids = np.array([i for i in kicids if i not in ebs])

        # Then the KOI false positives:
        kois = KOICatalog().df
        kois = kois[kois.koi_disposition == "FALSE POSITIVE"]
        kois = set(np.array(kois.kepid))
        kicids = np.array([i for i in kicids if i not in kois])

        print("Removed {0} known EBs from target list".format(l0-len(kicids)))

    # Check and create the output directory.
    if os.path.exists(args.output_dir):
        logging.warning("Output directory '{0}' exists"
                        .format(args.output_dir))
    else:
        os.makedirs(args.output_dir)
    cand_fn = os.path.join(args.output_dir, "candidates.csv")
    models = ["gp", "outlier", "box1", "box2", "step", "transit"]
    columns = [
        "kicid", "num_peaks", "peak_id",
        "accept_bic", "accept_time",
        "channel", "skygroup", "module", "output", "quarter", "season",
        "chunk", "t0", "s2n", "bkg", "depth", "depth_ivar",
        "transit_ror", "transit_duration", "transit_time",
        "chunk_min_time", "chunk_max_time",
    ]
    columns += ["lnlike_{0}".format(k) for k in models]
    columns += ["bic_{0}".format(k) for k in models]
    if args.inject:
        columns += ["injected_{0}".format(k) for k in ["t0", "period", "ror",
                                                       "b", "e", "omega"]]
        columns += ["recovered"]
    with open(cand_fn, "w") as f:
        f.write("# {0}\n".format(", ".join(columns)))
    with open(os.path.join(args.output_dir, "targets.txt"), "w") as f:
        f.write("\n".join(map("{0}".format, kicids)))

    if len(kicids):
        # Deal with parallelization.
        if args.parallel:
            pool = Pool()
            M = pool.imap_unordered
        else:
            M = map

        for peaks in tqdm(M(function, kicids), total=len(kicids)):
            if not len(peaks):
                continue
            with open(cand_fn, "a") as f:
                f.write("\n".join(
                    ", ".join("{0}".format(p.get(k, np.nan)) for k in columns)
                    for p in peaks) + "\n")

    if args.filenames is not None:
        lcs = load_light_curves(
            args.filenames,
            detrend_hw=args.detrend_hw,
            remove_kois=not args.no_remove_kois,
        )
        peaks = function(lcs=lcs)
        if len(peaks):
            with open(cand_fn, "a") as f:
                f.write("\n".join(
                    ", ".join("{0}".format(p.get(k, np.nan)) for k in columns)
                    for p in peaks) + "\n")
