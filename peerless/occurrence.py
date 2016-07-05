# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from itertools import product
from collections import OrderedDict

from peerless.catalogs import TargetCatalog

__all__ = ["compute_occurrence"]

# Newton's constant in $R_\odot^3 M_\odot^{-1} {days}^{-2}$.
_G = 2945.4625385377644


def compute_occurrence(inj, fits0, rgrid, pgrid):
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
    inj0 = pd.DataFrame(inj0[np.isfinite(inj0.mass)])

    # Note this only works because we injected uniformly in log-period.
    results = []
    for (rmin, rmax), (pmin, pmax) in product(rbins, pbins):
        m = (rmin * 0.0995 <= inj0.radius) & (inj0.radius <= rmax * 0.0995)
        m &= (pmin * 365.25 <= inj0.period) & (inj0.period <= pmax * 365.25)
        inj = pd.DataFrame(inj0[m])
        rec = inj.accept

        # Geometric.
        fk = inj.dutycycle
        Tk = inj.dataspan
        Mk = inj.mass
        Rk = inj.radius_stlr
        Qk = (4*np.pi/(_G * Mk))**(1./3) * (Rk + inj.radius * 0.0995)
        period = inj.period

        # Eccentric contribution to geometric transit probability.
        eccen = inj.e
        omega = inj.omega
        Qe = (1.0 + eccen * np.sin(omega)) / (1.0 - eccen**2)

        # Window function.
        Qt = period**(-2./3) * (1.0 - (1.0 - fk)**(Tk / period))

        # Re-weight the samples based on the injected radius distribution.
        radius = inj.radius
        bins = np.log(0.0995) + np.linspace(np.log(rmin), np.log(rmax), 10)
        n, _ = np.histogram(np.log(np.array(radius)), bins, density=True)
        w = n[np.digitize(np.log(np.array(radius)), bins) - 1]

        # Combine all the effects and integrate.
        Q = (Qk * Qt * Qe * w)[rec].sum() / w.sum()

        m = ((pmin < fits0.period) & (fits0.period < pmax) &
            (rmin < fits0.radius) & (fits0.radius < rmax))
        fits = fits0[m]

        N = len(fits)
        int_rate = N / (Q * len(targets))
        vol = (np.log(pmax) - np.log(pmin)) * (np.log(rmax) - np.log(rmin))

        results.append(OrderedDict([
            ("radius_min", rmin),
            ("radius_max", rmax),
            ("period_min", pmin),
            ("period_max", pmax),
            ("count", len(fits)),
            ("volume", vol),
            ("normalization", Q),
            ("rate_density", int_rate/vol),
            ("rate_density_uncert", int_rate/vol/np.sqrt(N)),
        ]))

    return pd.DataFrame(results)
