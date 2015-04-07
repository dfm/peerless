# -*- coding: utf-8 -*-

__all__ = ["TEXP", "PEERLESS_DATA_DIR"]

import os

TEXP = 1626.0 / 60. / 60. / 24.
PEERLESS_DATA_DIR = os.environ.get("PEERLESS_DATA_DIR",
                                   os.path.expanduser(
                                       os.path.join("~", "peerless")))
