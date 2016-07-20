peerless
========

The search for the transits of long-period exoplanets and binary star systems
in the archival *Kepler* light curves.


Running the code
----------------

This project is (in theory) reproducible given enough compute time. Here are
the steps:

Set up the environment & build extensions
+++++++++++++++++++++++++++++++++++++++++

To get started, you'll need to install `miniconda3
<https://www.continuum.io/downloads>`_ and then install the ``peerless``
environment:

.. code-block:: bash

    conda env create -f environment.yml
    source activate peerless

where ``environment.yml`` is in the root of this repository.

There are two packages that you'll need to install following the specific
installation instructions in their documentation: *(a)* the 1.0-dev branch of
`george <https://github.com/dfm/george>`_, and *(b)* `transit
<https://github.com/dfm/transit>`_.

Once this environment is enabled, set the environment variable:

.. code-block:: bash

    export PEERLESS_DATA_DIR="/path/to/scratch/"

to the directory where you want peerless to save all of its output. You'll
need something like a TB of disk space to run the full pipeline.

Then, you'll need to build the peerless extensions:

.. code-block:: bash

    python setup.py build_ext --inplace


Target selection & data download
++++++++++++++++++++++++++++++++

Next up, run the target selection and download all the relevant datasets:

.. code-block:: bash

    scripts/peerless-targets
    scripts/peerless-datasets -p {ncpu}
    scripts/peerless-download -p {ncpu}

where ``{ncpu}`` is the number of CPUs that you want to run in parallel using
``multiprocessing`` (they must be on the same node).


Transit search & injection tests
++++++++++++++++++++++++++++++++

To search these targets for transits, run:

.. code-block:: bash

    scripts/peerless-search -p {ncpu} -q --no-plots -o {searchdir}

where ``{ncpu}`` is the same as above and ``{searchdir}`` is the root
directory for the output.

Then to run a single pass of injection tests (one per target), run:

.. code-block:: bash

    scripts/peerless-search -p {ncpu} -q --no-plots --inject -o {injdir}/{someinteger}

Since you'll want to run many rounds of this script, the output directory
should be something like ``/path/to/injections/{someinteger}`` where
``{someinteger}`` is an integer identifying the run.

To collect the results of the search and injection tests, run:

.. code-block:: bash

    scripts/peerless-collect {searchdir} {injdir} -o {resultsdir}

where ``{searchdir}`` and ``{injdir}`` are from above and ``{resultsdir}`` is
the location where these should be saved.

False positive simulations & analysis
+++++++++++++++++++++++++++++++++++

Run the `predictions notebook <https://github.com/dfm/peerless/blob/master/prediction/prediction.ipynb>`_.  
Dependencies are `exosyspop <github.com/timothydmorton/exosyspop>`_, which
further depends on `isochrones <github.com/timothydmorton/exosyspop>`_ and `vespa <githubcom/timothydmorton/vespa>`_.

License
-------

Copyright 2015-2016 Daniel Foreman-Mackey

Licensed under the terms of the MIT License (see LICENSE).
