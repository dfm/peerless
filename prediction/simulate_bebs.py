#!/usr/bin/env python
from __future__ import print_function, division

import os, sys
sys.path.append('..')

sim_num = sys.argv[1]

from peerless.settings import PEERLESS_DATA_DIR

SIM_DIR = os.path.join(PEERLESS_DATA_DIR,'population_sims','bebs')

if not os.path.exists(SIM_DIR):
    os.makedirs(SIM_DIR)

import pandas as pd
import numpy as np

try:
    targets = pd.read_hdf('targets.h5', 'df')
except:
    from targets import targets
    targets.to_hdf('targets.h5', 'df')

pA = [-2.5038e-3, 0.12912, -2.4273, 19.980, -60.931]
pB = [3.0668e-3, -0.15902, 3.0365, -25.320, 82.605]
pC = [-1.5465e-5, 7.5396e-4, -1.2836e-2, 9.6434e-2, -0.27166]


def Pbg(Kp, b, r=4):
    """Expected number of BG stars within r", in magnitude range (Kp, Kp + 10)
    """
    if Kp < 11:
        Kp = 11
    if Kp > 16:
        Kp = 16
    return (r/2)**2*(np.polyval(pC, Kp) + 
                     np.polyval(pA, Kp)*np.exp(-b/np.polyval(pB, Kp)))


from astropy.coordinates import SkyCoord
from scipy.stats import poisson
from astropy import constants as const

G = const.G.cgs.value
MSUN = const.M_sun.cgs.value
RSUN = const.R_sun.cgs.value

from vespa.stars.populations import BGStarPopulation_TRILEGAL

ra, dec = 290.665452, 44.484019 #Center of Kepler field
bgpop = BGStarPopulation_TRILEGAL('mod13_bg.h5',ra=ra, dec=dec)

def generate_bg_targets(bg):

    c = SkyCoord(targets.ra, targets.dec, unit='deg')
    bs = c.galactic.b.deg

    dataspan = []
    dutycycle = []
    mass_A = []
    radius_A = []
    age = []
    feh = []
    kepmag_target = []
    kepmag_A = []
    index = []
    for (ix, s), b in zip(targets.iterrows(), bs):
        n = poisson(Pbg(s.kepmag, b)).rvs()
        for i in range(n):
            bg_star = bg.stars.ix[np.random.randint(len(bg.stars))]
            Kp = s.kepmag
            while not ((bg_star.Kepler_mag > Kp) and 
                       (bg_star.Kepler_mag < Kp+10)):
                bg_star = bg.stars.ix[np.random.randint(len(bg.stars))]
            mass = bg_star.Mact
            radius = np.sqrt(G * mass * MSUN / 10**bg_star.logg)/RSUN
            dataspan.append(s.dataspan)
            dutycycle.append(s.dutycycle)
            mass_A.append(mass)
            radius_A.append(radius)
            kepmag_target.append(s.kepmag)
            kepmag_A.append(bg_star.Kepler_mag)
            age.append(bg_star.logAge)
            feh.append(bg_star['[M/H]'])
            index.append(ix)

    bg_targets = pd.DataFrame({'dataspan':dataspan,
                              'dutycycle':dutycycle,
                              'mass_A':mass_A,
                              'radius_A':radius_A,
                               'age':age,
                               'feh':feh,
                               'kepmag_target':kepmag_target,
                               'kepmag_A':kepmag_A}, index=index)
    return bg_targets

bg_targets = generate_bg_targets(bgpop)

from sims import BG_BinaryPopulation
bgs = BG_BinaryPopulation(bg_targets)
bg_obs = bgs.observe(query='period > 5')

bg_obs.to_hdf(os.path.join(SIM_DIR,'{}.h5'.format(sim_num)), 'df')
print('{0} BEBs observed with P > 5d'.format(len(bg_obs)))
