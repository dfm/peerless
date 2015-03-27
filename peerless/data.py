# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["load_light_curves_for_kic", "load_light_curves"]

import fitsio
import numpy as np
from scipy.ndimage.measurements import label as contig_label

from .settings import HALF_WIDTH


def load_light_curves_for_kic(kicid, pdc=True, min_break=5, **kwargs):
    import kplr
    client = kplr.API()
    kwargs["fetch"] = kwargs.get("fetch", True)
    kwargs["short_cadence"] = kwargs.get("short_cadence", False)
    kwargs["order"] = kwargs.get("order", "sci_data_quarter")
    lcs = client.star(kicid).get_light_curves(**kwargs)
    return load_light_curves((lc.filename for lc in lcs), pdc=pdc,
                             min_break=min_break)


def load_light_curves(fns, pdc=True, min_break=1):
    lcs = []
    for fn in fns:
        # Load the data.
        data = fitsio.read(fn)
        x = data["TIME"]
        q = data["SAP_QUALITY"]
        if pdc:
            y = data["PDCSAP_FLUX"]
        else:
            y = data["SAP_FLUX"]

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

        # Remove bad quality points.
        y[q != 0] = np.nan

        # Find and flag long sections of missing NaNs.
        lbls, count = contig_label(~np.isfinite(y))
        for i in range(count):
            m = lbls == i+1
            # Label sections of missing fluxes longer than min_break points
            # by setting the times equal to NaN.
            if m.sum() > min_break:
                x[m] = np.nan

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
            lcs.append(LightCurve(x0, y0, meta))

    # Only retain chunks that are long enough (wrt the window half width).
    return [lc for lc in lcs if len(lc) > 2 * HALF_WIDTH]


class LightCurve(object):

    def __init__(self, time, flux, meta):
        self.time = np.ascontiguousarray(time, dtype=float)
        self.flux = np.ascontiguousarray(flux / np.median(flux), dtype=float)
        self.meta = meta
        self.footprint = self.time.max() - self.time.min()

    def __len__(self):
        return len(self.time)
