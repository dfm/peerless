# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["load_light_curves"]

import fitsio
import numpy as np

from .settings import HALF_WIDTH


def load_light_curves(fns):
    lcs = []
    for fn in fns:
        # Load the data.
        data = fitsio.read(fn)
        x = data["TIME"]
        y = data["PDCSAP_FLUX"]

        # Split into months.
        m = np.isfinite(x)
        gi = np.arange(len(x))[m]
        bi = np.arange(len(x))[~m]
        if len(bi):
            bi = bi[(bi > gi[0]) & (bi < gi[-1])]
            d = np.diff(bi)
            chunks = [slice(gi[0], bi[0])]
            for a, b in zip(bi[:-1][d > 1], bi[1:][d > 1]):
                chunks.append(slice(a+1, b-1))
            chunks.append(slice(bi[-1]+1, gi[-1]))
        else:
            chunks = [slice(gi[0], gi[-1])]

        # Interpolate missing data.
        for c in chunks:
            x0, y0 = x[c], y[c]
            m = np.isfinite(y0)
            if not np.any(m):
                continue
            y0[~m] = np.interp(x0[~m], x0[m], y0[m])
            lcs.append(LightCurve(x0, y0))

    # Only retain chunks that are long enough (wrt the window half width).
    return [lc for lc in lcs if len(lc) > 2 * HALF_WIDTH]


class LightCurve(object):

    def __init__(self, time, flux):
        self.time = time
        self.flux = flux / np.median(flux)

    def __len__(self):
        return len(self.time)
