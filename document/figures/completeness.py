# -*- coding: utf-8 -*-

from __future__ import division, print_function

from plot_setup import COLORS

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

inj = pd.read_hdf("../../results/injections.h5", "injections")
print(inj)


# fig.savefig("completeness.pdf", bbox_inches="tight")
