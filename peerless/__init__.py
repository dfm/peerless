# -*- coding: utf-8 -*-

__version__ = "0.0.0.dev0"

try:
    __PEERLESS_SETUP__  # NOQA
except NameError:
    __PEERLESS_SETUP__ = False

if not __PEERLESS_SETUP__:
    __all__ = [
        "data",
        "catalogs",
        "settings",
    ]
    from . import data, catalogs, settings
