# -*- coding: utf-8 -*-

from __future__ import division, print_function

from plot_setup import COLORS

import peerless
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl


x = np.linspace(0, 10, 500)
pl.plot(x, 0.1*x)
pl.plot(x, np.sin(x))
pl.savefig("blah.png")
