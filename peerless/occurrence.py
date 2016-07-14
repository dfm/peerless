# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from itertools import product
from collections import OrderedDict

from peerless.catalogs import TargetCatalog

__all__ = ["compute_occurrence"]

# Newton's constant in $R_\odot^3 M_\odot^{-1} {days}^{-2}$.
_G = 2945.4625385377644


def compute_occurrence(inj, fits0, pgrid, rgrid,
                       x_name="period", x_factor=365.25,
                       y_name="radius", y_factor=0.0995):
    """
    Compute the occurrence rate density in a grid of period

    """
    rbins = [(float(rgrid[i]), float(rgrid[i+1])) for i in range(len(rgrid)-1)]
    pbins = [(float(pgrid[i]), float(pgrid[i+1])) for i in range(len(pgrid)-1)]

    # Only select injections from the corresponding target list.
    targets = TargetCatalog().df
    targets = targets[np.isfinite(targets.mass)]
    inj0 = pd.merge(inj, targets, left_on="kicid", right_on="kepid",
                    suffixes=["", "_stlr"])
    inj0 = pd.DataFrame(inj0[np.isfinite(inj0.stlr_mass)])

    # Note this only works because we injected uniformly in log-period.
    results = []
    for (rmin, rmax), (pmin, pmax) in product(rbins, pbins):
        m = (rmin*y_factor <= inj0[y_name]) & (inj0[y_name] <= rmax*y_factor)
        m &= (pmin*x_factor <= inj0[x_name]) & (inj0[x_name] <= pmax*x_factor)
        inj = pd.DataFrame(inj0[m])
        rec = inj.accept

        # Geometric.
        fk = inj.dutycycle
        Tk = inj.dataspan
        Mk = inj.stlr_mass
        Rk = inj.stlr_radius
        period = inj.period
        Qk = period**(-2./3) * (4*np.pi/(_G * Mk))**(1./3) * (Rk + inj.radius)

        # Eccentric contribution to geometric transit probability.
        eccen = inj.e
        omega = inj.omega
        Qe = (1.0 + eccen * np.sin(omega)) / (1.0 - eccen**2)

        # Window function.
        Qt = (1.0 - (1.0 - fk)**(Tk / period))

        # Re-weight the samples based on the injected radius distribution.
        x_samp = np.log(np.array(inj[x_name]))
        x_bins = np.log(x_factor) + np.linspace(np.log(pmin), np.log(pmax), 5)
        y_samp = np.log(np.array(inj[y_name]))
        y_bins = np.log(y_factor) + np.linspace(np.log(rmin), np.log(rmax), 6)
        n, _, _ = np.histogram2d(x_samp, y_samp, (x_bins, y_bins))
        w = (1./n)[np.digitize(x_samp, x_bins) - 1,
                   np.digitize(y_samp, y_bins) - 1]

        # Combine all the effects and integrate.
        Q = (Qk * Qt * Qe * w)[rec].sum() / w.sum()

        m = ((pmin < fits0[x_name]) & (fits0[x_name] < pmax) &
            (rmin < fits0[y_name]) & (fits0[y_name] < rmax))
        fits = fits0[m]

        N = len(fits)
        int_rate = N / (Q * len(targets))
        vol = (np.log(pmax) - np.log(pmin)) * (np.log(rmax) - np.log(rmin))

        results.append(OrderedDict([
            (y_name + "_min", rmin),
            (y_name + "_max", rmax),
            (x_name + "_min", pmin),
            (x_name + "_max", pmax),
            ("count", len(fits)),
            ("volume", vol),
            ("normalization", Q),
            ("rate_density", int_rate/vol),
            ("rate_density_uncert", int_rate/vol/np.sqrt(N)),
        ]))

    return pd.DataFrame(results)
