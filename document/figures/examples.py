#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = []

from plot_setup import COLORS
from matplotlib import rcParams
rcParams["font.size"] = 16

import numpy as np
import matplotlib.pyplot as pl

import transit
import peerless

fns = [
    "/Users/dfm/peerless/lcs/kplr010287723-2009350155506_llc.fits",
]
lcs = peerless.data.load_light_curves(fns, min_break=50)
lc = lcs[np.argmax(map(len, lcs))]

m = np.isfinite(lc.time) & np.isfinite(lc.flux)
x, y1 = lc.time[m], lc.flux[m]
x -= np.mean(x)

system = transit.System(transit.Central())
system.add_body(transit.Body(r=0.04, period=1500., t0=np.mean(x)))
y2 = y1 * system.light_curve(x, texp=lc.texp)

fig, axes = pl.subplots(1, 2, sharey=True, sharex=True, figsize=(8, 3))
axes[0].plot(x, y1, ".k")
axes[1].plot(x, y2, ".k")

axes[0].set_xlim(-1.8, 1.8)
axes[0].yaxis.set_major_locator(pl.MaxNLocator(5))
axes[0].xaxis.set_major_locator(pl.MaxNLocator(4))
axes[0].set_yticklabels([])
axes[0].set_xlabel("time [days]")
axes[1].set_xlabel("time [days]")

fig.savefig("transit-vs-no.pdf")
