# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["COLORS"]

from matplotlib import rcParams
rcParams["font.size"] = 20
rcParams["font.family"] = "serif"
rcParams["font.serif"] = "Computer Sans"
rcParams["text.usetex"] = True
rcParams["figure.autolayout"] = True

import sys
sys.path.insert(0, "../..")
import peerless  # NOQA

COLORS = dict(
    DATA="k",
)

SQUARE_FIGSIZE = (4, 4)
