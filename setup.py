#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import glob
import numpy
from Cython.Build import cythonize
from setuptools import setup, Extension

# Set up the extension.
kwargs = dict(
    include_dirs=[numpy.get_include()],
    extra_compile_args=["-Wno-unused-function", ],
)
exts = [
    Extension("peerless._search", sources=["peerless/_search.pyx"],
              **kwargs),
]

# Hackishly inject a constant into builtins to enable importing of the
# package before the library is built.
if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins
builtins.__PEERLESS_SETUP__ = True
import peerless  # NOQA

# Execute the setup command.
desc = open("README.rst").read()
setup(
    name="peerless",
    version=peerless.__version__,
    author="Daniel Foreman-Mackey",
    author_email="foreman.mackey@gmail.com",
    packages=[
        "peerless",
    ],
    ext_modules=cythonize(exts),
    scripts=list(glob.glob(os.path.join("scripts", "peerless-*"))),
    url="http://github.com/dfm/peerless",
    license="MIT",
    description="I can haz planetz?",
    long_description=desc,
    package_data={"": ["README.rst", "LICENSE"],
                  "peerless": ["data/*.csv"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python 3",
    ],
)
