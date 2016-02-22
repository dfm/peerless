peerless
========

The search for the transits of long-period exoplanets and binary star systems
in the archival *Kepler* light curves.


Running the code
----------------

This project is (in theory) reproducible given enough compute time. Here are
the steps:


1. To get started, you'll need to install `miniconda3
<https://www.continuum.io/downloads>`_ and then install the ``peerless``
environment:

.. code-block:: bash

    conda env create -f environment.yml
    source activate peerless

where ``environment.yml`` is in the root of this repository.

2. There are two packages that you'll need to install following the specific
installation instructions in their documentation: *(a)* the 1.0-dev branch of
`george <https://github.com/dfm/george>`_, and *(b)* `transit
<https://github.com/dfm/transit>`_.

3. Once this environment is enabled, set the environment variable:

.. code-block:: bash

    export PEERLESS_DATA_DIR="/path/to/scratch/"

to the directory where you want peerless to save all of its output. You'll
need something like a TB of disk space to run the full pipeline.

4. Then, you'll need to build the peerless extensions:

.. code-block:: bash

    python setup.py build_ext --inplace



License
-------

Copyright 2015-2016 Daniel Foreman-Mackey

Licensed under the terms of the MIT License (see LICENSE).
