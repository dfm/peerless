#!/usr/bin/env python
from __future__ import print_function, division

import os, sys
sys.path.append('..')

sim_num = sys.argv[1]

from peerless.settings import PEERLESS_DATA_DIR

SIM_DIR = os.path.join(PEERLESS_DATA_DIR,'population_sims','ebs')

if not os.path.exists(SIM_DIR):
    os.makedirs(SIM_DIR)

import pandas as pd

try:
    targets = pd.read_hdf('targets.h5', 'df')
except:
    from targets import targets
    targets.to_hdf('targets.h5', 'df')

# Simulation a population of binary stars and observe them.

from sims import BinaryPopulation

ebs = BinaryPopulation(targets)
eb_obs = ebs.observe(query='period > 5')

eb_obs.to_hdf(os.path.join(SIM_DIR,'{}.h5'.format(sim_num)), 'df')
print('{0} EBs observed with P > 5d'.format(len(eb_obs)))
