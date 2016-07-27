# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import pandas as pd

import matplotlib.pyplot as pl

from peerless.plot_setup import SQUARE_FIGSIZE

f_duty = 0.7
T_k = 700.0

P = np.linspace(100.0, 5000.0, 5000)

fig, ax = pl.subplots(1, 1, figsize=SQUARE_FIGSIZE)
ax.plot(P, 1.0 - (1.0 - f_duty) ** (T_k / P), "k")
ax.plot(P, (T_k * f_duty) / P, "g")
ax.xaxis.set_major_locator(pl.MaxNLocator(3))
ax.set_ylim(0, 1.0)
ax.set_xlabel("period")
ax.set_ylabel("$Q_t$")
fig.savefig("window_function.png")
