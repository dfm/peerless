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

    def map(self, function, arglist, dirlist):
        # Get a view into the pool.
        view = self.client[:]

        # Loop over the argument list and execute the jobs.
        jobs = []
        for args, bp in zip(arglist, dirlist):
            # Make sure that the path exists.
            if not os.path.exists(bp):
                os.makedirs(bp)

            # Build the function wrapper.
            f = _wrapper(function, bp)

            # Apply the job on the client.
            jobs.append((bp, view.apply(f, args)))

        # Monitor the jobs and check for completion and errors.
        retrieved = [False] * len(jobs)
        while not all(retrieved):
            for i, (fn, j) in enumerate(jobs):
                if j.ready() and not retrieved[i]:
                    try:
                        j.get()
                    except Exception:
                        with open(os.path.join(bp, "error.log"), "a") as f:
                            f.write("Uncaught error:\n\n")
                            f.write(traceback.format_exc())
                    else:
                        with open(os.path.join(bp, "success.log"), "w") as f:
                            f.write("Finished at: {0}\n".format(time.time()))
                    retrieved[i] = True
            time.sleep(3)


class _wrapper(object):

    def __init__(self, function, bp, name="output.log", error="error.log"):
        self.filename = os.path.join(bp, name)
        self.error = os.path.join(bp, error)
        self.function = function

    def __call__(self, args):
        try:
            strt = time.time()
            with Capturing(self.filename):
                result = self.function(args)
                print("Execution time: {0} seconds".format(time.time()-strt))
            return result
        except Exception:
            with open(self.error, "a") as f:
                f.write("Error during execution:\n\n")
                f.write(traceback.format_exc())
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
