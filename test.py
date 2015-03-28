#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

from peerless.pool import IPythonPool


def blah(x):
    print(x)
    if x == 0:
        assert 0
    return x**2


pool = IPythonPool()
pool.client[:].push(dict(blah=blah), block=True)
r = pool.run(blah, range(5), map("out/{0}".format, range(5)), quiet=True)
print(r)
