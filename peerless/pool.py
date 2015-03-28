# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["IPythonPool"]

import os
import sys
import time
import traceback
from IPython.parallel import Client


class IPythonPool(object):

    def __init__(self, **kwargs):
        self.client = Client(**kwargs)

    def run(self, function, arglist, dirlist, quiet=False, **kwargs):
        f = _wrapper(function, **kwargs)
        return self.client[:].map(f, zip(dirlist, arglist))


class _wrapper(object):

    def __init__(self, function, name="output.log", error="error.log",
                 **kwargs):
        self.filename = name
        self.error = error
        self.function = function
        self.kwargs = kwargs

    def __call__(self, args):
        bp = args[0]
        if not os.path.exists(bp):
            os.makedirs(bp)
        try:
            strt = time.time()
            with Capturing(os.path.join(bp, self.filename)):
                result = self.function(args[1], **(self.kwargs))
                print("Execution time: {0} seconds".format(time.time()-strt))
            return result
        except Exception:
            with open(os.path.join(bp, self.error), "a") as f:
                f.write(traceback.format_exc() + "\n\n")
            raise


# Insane hackish output capturing context.
# http://stackoverflow.com/questions/16571150
#   /how-to-capture-stdout-output-from-a-python-function-call
class Capturing(object):

    def __init__(self, fn):
        self.fn = fn

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._fh = open(self.fn, "a")
        return self

    def __exit__(self, *args):
        self._fh.close()
        sys.stdout = self._stdout
