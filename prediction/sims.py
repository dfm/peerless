from __future__ import print_function, division

import numpy as np

from isochrones.dartmouth import Dartmouth_Isochrone
from vespa.stars.utils import draw_eccs # this is a function that returns
                                        # empirically reasonably eccentricities
                                        # for given binary periods.
from scipy.stats import norm, binom
from vespa.transit_basic import _quadratic_ld

from vespa.stars.utils import G, MSUN, RSUN, AU, DAY

def draw_powerlaw(alpha, rng, N=1):
    """
    Returns random variate according to x^alpha, between rng[0] and rng[1]
    """
    if alpha == -1:
        alpha = -1.0000001
    # Normalization factor
    x0, x1 = rng
    C = (alpha + 1) / (x1**(alpha + 1) - x0**(alpha + 1))
    
    if N==1:
        u = np.random.random()
    else:
        u = np.random.random(N)
    x = ((u * (alpha + 1)) / C + x0**(alpha + 1))**(1./(alpha + 1))

    return x

def semimajor(P,mstar=1):
    """Returns semimajor axis in AU given P in days, mstar in solar masses.
    """
    return ((P*DAY/2/np.pi)**2*G*mstar*MSUN)**(1./3)/AU


class BinaryPopulation(object):
    """
    Initialized with population of primary stars

    """
    #parameters for binary population (for period in years)
    param_names = ['fB', 'gamma', 'qmin', 'mu_logp', 'sig_logp']
    default_params = [0.4, 0.3, 0.1, np.log10(250), 2.3]

    # Physical and orbital parameters that can be accessed.
    physical_props = ['mass_A', 'radius_A',
                      'mass_B', 'radius_B', 'flux_ratio']

    orbital_props = ['period', 'ecc', 'w', 'inc', 'a',
                      'b_pri', 'b_sec', 'k', 'tra', 'occ',
                      'd_pri', 'd_sec', 'T14_pri', 'T14_sec']
                     
    # property dictionary mapping to DataFrame column
    # Default here is for KICatalog names
    prop_columns = {'mass_A': 'mass', 'radius_A': 'radius'}

    def __init__(self, stars, params=None, band='Kepler', 
                 ic=Dartmouth_Isochrone):

        self.stars = stars
        self.band = band
        self._ic = ic
        self._params = params

        for k,v in self.prop_columns.items():
            self.stars.loc[:, k] = self.stars.loc[:, v]

    @property
    def params(self):
        if self._params is not None:
            return self._params
        else:
            return self.default_params

    @property
    def ic(self):
        if type(self._ic)==type:
            self._ic = self._ic()
        return self._ic

    @property
    def N(self):
        return len(self.stars)

    def __getattr__(self, name):
        if name in self.stars:
            return self.stars[name].values
        elif name in self.physical_props:
            if name not in self.stars:
                self._generate_binaries()
        elif name in self.orbital_props:
            if name not in self.stars:
                self._generate_orbits()
        return self.stars[name].values

    def _generate_binaries(self):
        N = self.N
        fB, gamma, qmin, _, _ = self.params

        b = np.random.random(N) < fB
        
        # Simulate secondary masses
        minmass = self.ic.minmass
        qmin = np.maximum(qmin, minmass/self.mass_A)
        q = draw_powerlaw(gamma, (qmin, 1), N=N)
        M2 = (q * self.mass_A)[b]

        # Stellar catalog doesn't have ages, so let's make them up.
        #  ascontiguousarray makes ic calls faster.
        ic = self.ic
        feh = np.ascontiguousarray(np.clip(self.feh, ic.minfeh, ic.maxfeh))
        minage, maxage = ic.agerange(self.mass_A, feh)
        maxage = np.clip(maxage, 0, ic.maxage)
        if 'age' not in self.stars:
            minage += 0.3 # stars are selected to not be active
            maxage -= 0.1
            age = np.random.random(size=N) * (maxage - minage) + minage
        else:
            age = np.clip(self.stars.age.values, minage, maxage)

        # Secondary properties (don't let secondary be bigger than primary)
        M2 = np.ascontiguousarray(M2)
        R2 = ic.radius(M2, age[b], feh[b])
        R1 = self.radius_A[b]
        toobig = R2 > R1
        R2[toobig] = R1[toobig]

        # Calculate secondary/primary flux ratio
        M1 = np.ascontiguousarray(self.mass_A[b])
        dmag = (ic.mag[self.band](M2, age[b], feh[b]) - 
                ic.mag[self.band](M1, age[b], feh[b]))
        flux_ratio = 10**(-0.4 * dmag)

        # Assign columns appropriately.  
        stars = self.stars.copy()

        stars.loc[b, 'mass_B'] = M2
        stars.loc[b, 'radius_B'] = R2
        stars.loc[b, 'flux_ratio'] = flux_ratio
        stars.loc[:, 'age'] = age
        
        self.stars = stars

    def _generate_orbits(self, p=None):
        _, _, _, mu_logp, sig_logp = self.params

        N = self.N

        period = 10**(norm(np.log10(mu_logp), sig_logp).rvs(N)) * 365.25
        ecc = draw_eccs(N, period)
        w = np.random.random(N) * 2 * np.pi
        inc = np.arccos(np.random.random(N))        
        a = semimajor(period, self.mass_A + self.mass_B) * AU

        # Determine closest approach
        b_pri = a*np.cos(inc)/(self.radius_A*RSUN) * (1-ecc**2)/(1 + ecc*np.sin(w))
        b_sec = a*np.cos(inc)/(self.radius_A*RSUN) * (1-ecc**2)/(1 - ecc*np.sin(w))

        R_tot = self.radius_A + self.radius_B
        tra = (b_pri < R_tot)
        occ = (b_sec < R_tot)

        # Calculate eclipse depths, assuming Solar limb darkening for all
        d_pri = np.zeros(N)
        d_sec = np.zeros(N)
        k = self.radius_B / self.radius_A
        T14_pri = period/np.pi*np.arcsin(self.radius_A*RSUN/a * np.sqrt((1+k)**2 - b_pri**2)/np.sin(inc)) *\
            np.sqrt(1-ecc**2)/(1+ecc*np.sin(w))
        T14_sec = period/np.pi*np.arcsin(self.radius_A*RSUN/a * np.sqrt((1+k)**2 - b_sec**2)/np.sin(inc)) *\
            np.sqrt(1-ecc**2)/(1-ecc*np.sin(w))
    
        T14_pri[np.isnan(T14_pri)] = 0.
        T14_sec[np.isnan(T14_sec)] = 0.

        for i in xrange(N):
            if tra[i]:
                f = _quadratic_ld._quadratic_ld(np.array([b_pri[i]]), k[i], 0.394, 0.296, 1)[0]
                F2 = self.flux_ratio[i]
                d_pri[i] = 1 - (F2 + f)/(1+F2)
            if occ[i]:
                f = _quadratic_ld._quadratic_ld(np.array([b_sec[i]/k[i]]), 1./k[i], 0.394, 0.296, 1)[0]
                F2 = self.flux_ratio[i]
                d_sec[i] = 1 - (1 + F2*f)/(1+F2)

        stars = self.stars.copy()
        for var in self.orbital_props:
            stars.loc[:, var] = eval(var)

        self.stars = stars

    def observe(self, query=None):
        """
        Returns catalog of the following observable quantities:
          
          * n_pri
          * n_sec
          * d_pri
          * d_sec
          * T14_pri
          * T14_sec
       
        assumes stars dataframe has 'dataspan' and 'dutycycle' columns
        """
        
        # Select only systems with eclipsing (or occulting) geometry
        m = self.tra | self.occ
        if query is None:
            query = 'dataspan > 0'
        df = self.stars.loc[m].query(query).copy()

        # Phase of secondary (Hilditch (2001) , Kopal (1959))
        #  Primary is at phase=0
        X = np.pi + 2*np.arctan(df.ecc * np.cos(df.w) / np.sqrt(1-df.ecc**2))
        secondary_phase = X - np.sin(X)

        # Assign each system a random phase at t=0;
        N = len(df)
        initial_phase = np.random.random(N)
        final_phase = initial_phase + df.dataspan/df.period

        # Determine number of primary & secondary eclipses, assuming perfect duty cycle
        n_pri_ideal = np.floor(final_phase) * df.tra
        n_sec_ideal = (np.floor(final_phase + secondary_phase) - 
                       np.floor(initial_phase + secondary_phase))*df.occ

        # Correct for duty cycle.  
        # Each event has probability (1-dutycycle) of landing in a gap.
        n_pri = np.zeros(N)
        n_sec = np.zeros(N)
        for i, (n1,n2,d) in enumerate(zip(n_pri_ideal,
                                          n_sec_ideal,
                                          df.dutycycle)):
            n_pri[i] = binom(n1,d).rvs()
            n_sec[i] = binom(n2,d).rvs()
        
        df.loc[:, 'n_pri'] = n_pri
        df.loc[:, 'n_sec'] = n_sec

        return df.query('(n_pri > 0) or (n_sec > 0)')


class BG_BinaryPopulation(BinaryPopulation):
    prop_columns = {}
    
    def _generate_orbits(self, p=None):
        super(BG_BinaryPopulation, self)._generate_orbits(p=p)
        
        F_target = 10**(-0.4*self.stars.kepmag_target)
        F_A = 10**(-0.4*self.stars.kepmag_A)
        F_B = self.stars.flux_ratio*F_A
        frac = (F_A + F_B)/(F_A + F_B + F_target)
        self.d_pri *= frac
        self.d_sec *= frac
        
