#!/usr/bin/env python
# -*- coding: utf-8 -*-

from plot_setup import COLORS

import numpy as np
import matplotlib.pyplot as pl
from peerless.catalogs import (
    KICatalog, KOICatalog, EBCatalog, BlacklistCatalog
)


stlr = KICatalog().df
m = (4200 <= stlr.teff) & (stlr.teff <= 6100)
m &= stlr.radius <= 1.15
m &= stlr.dataspan > 365.25*2.
m &= stlr.dutycycle > 0.6
m &= stlr.rrmscdpp07p5 <= 1000.
m &= stlr.kepmag < 15.

# Remove known EBs.
ebs = set(np.array(EBCatalog().df["#KIC"]))

# Then the KOI false positives:
kois = KOICatalog().df
kois = kois[kois.koi_disposition == "FALSE POSITIVE"]
fps = set(np.array(kois.kepid))

# And then finally the blacklist.
bl = set(np.array(BlacklistCatalog().df.kicid))

# The full list of ignores.
ignore = ebs | fps | bl
m &= ~stlr.kepid.isin(ignore)

fig, ax = pl.subplots(1, 1, figsize=(4, 4))
ax.plot(stlr[~m].teff, stlr[~m].logg, ".", color=COLORS["DATA"], ms=2,
        alpha=1.0, rasterized=True)
ax.plot(stlr[m].teff, stlr[m].logg, ".", color=COLORS["MODEL_2"], ms=2,
        alpha=1.0, rasterized=True)
ax.set_xlim(8250, 3500)
ax.set_ylim(5.1, 3.5)
ax.set_ylabel("$\log g$")
ax.set_xlabel("$T_\mathrm{eff}$")
ax.xaxis.set_major_locator(pl.MaxNLocator(4))
ax.yaxis.set_major_locator(pl.MaxNLocator(5))
fig.set_tight_layout(True)
fig.savefig("targets.pdf", dpi=400)

print("Found {0} targets".format(m.sum()))

np.savetxt("targets.txt", np.array(stlr[m].kepid), fmt="%d")
