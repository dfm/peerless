import glob
from peerless.model import Model
from peerless.data import load_light_curves_for_kic

# fns = glob.glob("/Users/dfm/.kplr/data/lightcurves/004150804/*_llc.fits")
# fns = glob.glob("/Users/dfm/.kplr/data/lightcurves/002162635/*_llc.fits")
# fns = glob.glob("/Users/dfm/.kplr/data/lightcurves/000757099/*_llc.fits")
# fns = glob.glob("/Users/dfm/.kplr/data/lightcurves/003558849/*_llc.fits")
# fns = glob.glob("/Users/dfm/.kplr/data/lightcurves/012356617/*_llc.fits")
# fns = glob.glob("/Users/dfm/.kplr/data/lightcurves/005966097/*_llc.fits")
lcs = load_light_curves_for_kic(10748390)
print(lcs)
mod = Model(lcs, npos=20000, min_period=900, max_period=5000, dt=0.1)
