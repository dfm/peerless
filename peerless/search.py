# -*- coding: utf-8 -*-

__all__ = ["search"]

import os
import logging
import numpy as np
import matplotlib.pyplot as pl
from scipy.optimize import minimize
from collections import OrderedDict

import george
from george import kernels, ModelingMixin
from george.modeling import check_gradient

import transit

from peerless._search import search as cython_search
from peerless.data import load_light_curves_for_kic, running_median_trend


class SearchResults(object):

    def __init__(self,
                 kicid,
                 chunks,
                 duration,
                 detect_thresh,
                 search_time,
                 search_depth,
                 search_depth_ivar,
                 search_scalar,
                 search_background,
                 injection=None):
        self.kicid = kicid
        self.chunks = chunks

        self.duration = duration
        self.detect_thresh = detect_thresh

        self.search_time = search_time
        self.search_depth = search_depth
        self.search_depth_ivar = search_depth_ivar
        self.search_scalar = search_scalar
        self.search_background = search_background

        self.injection = injection

        self.peaks = []


def search(kicid_and_injection=None,
           lcs=None,
           tau=0.6,
           detrend_hw=2.0,
           remove_kois=True,
           grid_frac=0.25,
           noise_hw=15.0,
           detect_thresh=25.0,
           max_fit_data=500,
           max_peaks=3,
           min_datapoints=10,
           all_models=False,
           delete=True,
           verbose=False):
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
        Relative S/N detection threshold. (default: 25.0)

    :param max_fit_data:
        The maximum number of data points for fitting. (default: 500)

    :param max_peaks:
        The maximum number of peaks to analyze in detail. (default: 3)

    :param min_datapoints:
        The minimum number of in-transit data points. (default: 10)

    :param delete:
        Delete the light curve files after loading them. (default: True)

    :param verbose:
        Moar printing. (default: False)

    """
    if kicid_and_injection is not None:
        kicid, injection = kicid_and_injection
    else:
        kicid, injection = None, None
    inject = injection is not None

    system = None
    if inject:
        system = transit.System(transit.Central(q1=injection["q1"],
                                                q2=injection["q2"]))
        system.add_body(transit.Body(
            radius=injection["ror"],
            period=injection["period"],
            b=injection["b"],
            e=injection["e"],
            omega=injection["omega"],
            t0=injection["t0"],
        ))
        injection["ncadences"] = 0
        injection["recovered"] = False

    if lcs is None and kicid is None:
        raise ValueError("you must specify 'lcs' or 'kicid'")
    if lcs is None:
        lcs, ncad = load_light_curves_for_kic(kicid, delete=delete,
                                              detrend_hw=detrend_hw,
                                              remove_kois=remove_kois,
                                              inject_system=system)
        if inject:
            injection["ncadences"] += ncad

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
        d, ivar, s = cython_search(tau, time[-1], lc.time, lc.flux-1,
                                   1/lc.ferr**2)
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

    results = SearchResults(kicid, lcs, tau, detect_thresh,
                            time, depth, depth_ivar, s2n, noise,
                            injection=injection)

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
        return results

    for i, peak in enumerate(peaks):
        peak["num_peaks"] = len(peaks)
        peak["peak_id"] = i

    if verbose and len(peaks) > max_peaks:
        logging.warning("truncating peak list")
    peaks = peaks[:max_peaks]

    # For each peak, plot the diagnostic plots and vet.
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
        cen_x = np.ascontiguousarray(lc0.mom_cen_1[inds])
        cen_y = np.ascontiguousarray(lc0.mom_cen_2[inds])

        peak["data"] = (x, y, yerr, cen_x, cen_y)

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
        peak["gps"] = OrderedDict()
        peak["pred_cens"] = []
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
            peak["gps"][name] = (gp, y)

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

            # Initialize one of the boxes using the transit shape.
            if name == "transit":
                models["box1"].mn = system.t0 - 0.5*system.duration
                models["box1"].mx = system.t0 + 0.5*system.duration

                # Fit the centroids.
                depth = 1.0 - float(system.get_value(system.t0))
                tm = (1.0 - system.get_value(x)) / depth
                A = np.vander(tm, 2)
                AT = A.T

                offset = 0.0
                offset_err = 0.0
                for ind, c in enumerate((cen_x, cen_y)):
                    err = np.median(np.abs(np.diff(c)))
                    kernel = np.var(c) * kernels.Matern32Kernel(2**2)
                    gp = george.GP(kernel, white_noise=2*np.log(np.mean(err)),
                                   fit_white_noise=True,
                                   mean=CentroidModel(tm, a=0.0, b=0.0),
                                   fit_mean=True)
                    gp.compute(x, err)

                    r = minimize(gp.nll, gp.get_vector(), jac=gp.grad_nll,
                                 args=(c - np.mean(c),), method="L-BFGS-B")
                    gp.set_vector(r.x)

                    C = gp.get_matrix(x)
                    C[np.diag_indices_from(C)] += err**2
                    alpha = np.linalg.solve(C, A)
                    ATA = np.dot(AT, alpha)
                    ATA[np.diag_indices_from(ATA)] *= 1 + 1e-8
                    mu = np.mean(c)
                    a = np.linalg.solve(C, c - mu)
                    w = np.linalg.solve(ATA, np.dot(AT, a))

                    offset += w[0]**2
                    offset_err = np.linalg.inv(ATA)[0, 0] * w[0]**2
                    peak["pred_cens"].append(np.dot(A, w) + mu)

                offset_err = np.sqrt(offset_err / offset)
                offset = np.sqrt(offset)

                peak["centroid_offset"] = offset
                peak["centroid_offset_err"] = offset_err

            if (peak["bic_{0}".format(name)] > peak["bic_transit"]
                    and not all_models):
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
            peak["gps"]["outlier"] = (gp, y0)

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
        peak["transit_depth"] = 1.0 - float(system.get_value(system.t0))

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

        # Save the injected parameters.
        if inject:
            for k in ["t0", "period", "ror", "b", "e", "omega"]:
                peak["injected_{0}".format(k)] = injection[k]

            # Check for recovery.
            p = injection["period"]
            d = (peak["transit_time"] - injection["t0"] + 0.5*p) % p - 0.5*p
            peak["is_injection"] = np.abs(d) < peak["transit_duration"]
            results.injection["recovered"] |= accept

        # Save the peak.
        results.peaks.append(peak)

    return results


def plot(results, output_dir="output", plot_all=False):
    basedir = os.path.join(output_dir, "{0}".format(results.kicid))
    for i, peak in enumerate(results.peaks):
        accept = peak["accept_bic"] and peak["accept_time"]
        if (not accept) and (not plot_all):
            continue

        x, y, yerr, cen_x, cen_y = peak["data"]
        system = peak["gps"]["transit"][0].mean

        # Centroid plot.
        fig, axes = pl.subplots(3, 1, figsize=(7, 8), sharex=True)

        ax = axes[0]
        ax.plot(x, (y-1)*1e3, "k")
        ax.plot(x, (system.get_value(x)-1)*1e3, "r", lw=1.5)
        ax.set_ylabel("flux")

        ax = axes[1]
        mu_x = np.mean(cen_x)
        ax.plot(x, cen_x - mu_x, "k")
        ax.set_ylabel("x-centroid")

        ax = axes[2]
        mu_y = np.mean(cen_y)
        ax.plot(x, cen_y - mu_y, "k")
        if len(peak["pred_cens"]):
            axes[1].plot(x, peak["pred_cens"][0] - mu_x, "g")
            axes[2].plot(x, peak["pred_cens"][1] - mu_y, "g")
        ax.set_ylabel("y-centroid")
        ax.set_xlabel("time [KBJD]")

        for ax in axes:
            ax.axvline(system.t0, color="g", lw=5, alpha=0.3)
            ax.set_xlim(system.t0-5, system.t0+5)
            ax.yaxis.set_major_locator(pl.MaxNLocator(4))
            ax.xaxis.set_major_locator(pl.MaxNLocator(5))
            ax.yaxis.set_label_coords(-0.15, 0.5)

        fig.subplots_adjust(
            left=0.18, bottom=0.1, right=0.98, top=0.97,
            wspace=0.05, hspace=0.12
        )

        os.makedirs(basedir, exist_ok=True)
        fig.savefig(os.path.join(basedir, "cen-{0:04d}.png".format(i + 1)))
        pl.close(fig)

        # Plots.
        fig, axes = pl.subplots(3, 2, figsize=(10, 8))

        # Raw flux.
        row = axes[0]
        for ax in row:
            [ax.plot(lc.raw_time, (lc.raw_flux-1)*1e3, ".k")
             for lc in results.chunks]

            ax.set_xticklabels([])
            ax.yaxis.set_major_locator(pl.MaxNLocator(4))
            ax.xaxis.set_major_locator(pl.MaxNLocator(5))

        row[0].set_ylabel("raw [ppt]")
        ax = row[1]

        x0 = np.linspace(x.min(), x.max(), 500)
        for k in ["gp", "outlier"]:
            if k not in peak["gps"]:
                continue
            gp, y0 = peak["gps"][k]
            mu = gp.predict(y0, x0, return_cov=False)
            ax.plot(x0, (mu-1)*1e3, lw=1.5)
        for k in ["transit", "box1", "box2", "step"]:
            if k not in peak["gps"]:
                continue
            model = peak["gps"][k][0].mean
            ax.plot(x0, (model.get_value(x0)-1)*1e3, lw=1.5)

        # De-trended flux.
        row = axes[1]
        for ax in row:
            [ax.plot(lc.time, (lc.flux-1)*1e3, ".k") for lc in results.chunks]
            ax.set_xticklabels([])
            ax.yaxis.set_major_locator(pl.MaxNLocator(4))
            ax.xaxis.set_major_locator(pl.MaxNLocator(5))
        row[0].set_ylabel("de-trended [ppt]")

        # Periodogram.
        row = axes[2]
        tau = results.duration
        detect_thresh = results.detect_thresh
        time = results.search_time
        scalar = results.search_scalar
        bkg = results.search_background
        for ax in row:
            ax.plot(time + 0.5*tau, scalar, "k")
            ax.plot(time + 0.5*tau, bkg, "g")
            ax.plot(time + 0.5*tau, detect_thresh * bkg, ":g")
            ax.yaxis.set_major_locator(pl.MaxNLocator(4))
            ax.xaxis.set_major_locator(pl.MaxNLocator(5))
            ax.set_xlabel("time [KBJD]")
        row[0].set_ylabel("s/n")

        for ax1, ax2 in axes:
            ax1.yaxis.set_label_coords(-0.15, 0.5)

            ax1.set_xlim(time.min() - 5.0, time.max() + 5.0)
            ax1.axvline(system.t0, color="g", lw=5, alpha=0.3)

            ax2.set_xlim(system.t0 - 5.0, system.t0 + 5.0)
            ax2.axvline(system.t0, color="g", lw=5, alpha=0.3)
            ax2.axvline(system.t0 - 0.5*tau, color="k", ls="dashed")
            ax2.axvline(system.t0 + 0.5*tau, color="k", ls="dashed")
            ax2.set_yticklabels([])

        fig.subplots_adjust(
            left=0.15, bottom=0.1, right=0.98, top=0.97,
            wspace=0.05, hspace=0.12
        )
        fig.savefig(os.path.join(basedir, "{0:04d}.png".format(i + 1)))
        pl.close(fig)


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


class CentroidModel(ModelingMixin):

    def __init__(self, mdl, **kwargs):
        self.mdl = mdl
        super(CentroidModel, self).__init__(**kwargs)

    def get_value(self, t):
        return self.a + self.b * self.mdl

    @ModelingMixin.parameter_sort
    def get_gradient(self, t):
        return dict(
            a=np.ones_like(self.mdl),
            b=self.mdl,
        )
