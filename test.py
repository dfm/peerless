#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

from peerless.pool import IPythonPool


def blah(x):
    return x[0]**2


pool = IPythonPool()
r = pool.run(blah, [[i] for i in range(5)], map("out/{0}".format, range(5)))
print(r)
