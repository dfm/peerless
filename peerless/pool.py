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
        # Get a view into the pool.
        view = self.client[:]

        # Loop over the argument list and execute the jobs.
        jobs = []
        for args, bp in zip(arglist, dirlist):
            # Make sure that the path exists.
            if not os.path.exists(bp):
                os.makedirs(bp)

            # Build the function wrapper.
            f = _wrapper(function, bp, **kwargs)

            # Apply the job on the client.
            jobs.append((bp, view.apply(f, args)))

        # Monitor the jobs and check for completion and errors.
        retrieved = [False for i in range(len(jobs))]
        while not all(retrieved):
            for i, (bp, j) in enumerate(jobs):
                if j.ready() and not retrieved[i]:
                    try:
                        j.get()
                    except Exception:
                        if not quiet:
                            raise
                    retrieved[i] = True
            time.sleep(3)


class _wrapper(object):

    def __init__(self, function, bp, name="output.log", error="error.log",
                 **kwargs):
        self.filename = os.path.join(bp, name)
        self.error = os.path.join(bp, error)
        self.function = function
        self.kwargs = kwargs

    def __call__(self, args):
        try:
            strt = time.time()
            with Capturing(self.filename):
                result = self.function(args, **(self.kwargs))
                print("Execution time: {0} seconds".format(time.time()-strt))
            return result
        except Exception:
            with open(self.error, "a") as f:
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
