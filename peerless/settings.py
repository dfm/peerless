# -*- coding: utf-8 -*-

__all__ = ["PEERLESS_DATA_DIR"]

import os

PEERLESS_DATA_DIR = os.environ.get("PEERLESS_DATA_DIR",
                                   os.path.expanduser(
                                       os.path.join("~", "peerless")))
