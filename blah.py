import glob
from peerless.model import Model
from peerless.data import load_light_curves

fns = glob.glob("/Users/dfm/.kplr/data/lightcurves/004150804/*_llc.fits")
# fns = glob.glob("/Users/dfm/.kplr/data/lightcurves/002162635/*_llc.fits")
# fns = glob.glob("/Users/dfm/.kplr/data/lightcurves/000757099/*_llc.fits")
# fns = glob.glob("/Users/dfm/.kplr/data/lightcurves/003558849/*_llc.fits")
# fns = glob.glob("/Users/dfm/.kplr/data/lightcurves/012356617/*_llc.fits")
# fns = glob.glob("/Users/dfm/.kplr/data/lightcurves/005966097/*_llc.fits")
lcs = load_light_curves(fns)
mod = Model(lcs, npos=50000, min_period=150, max_period=500, dt=0.1)
