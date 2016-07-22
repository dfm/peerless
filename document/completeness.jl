using PyPlot
using CGS
nperiod = 100
p = logspace(log10(2*365.25),log10(5000),nperiod)
lnR = log(5*REARTH/RSUN)
qdet = zeros(nperiod)
for i=1:nperiod
  qdet[i] = minimum([maximum([(-0.06*log(p[i])+0.78)*(-0.45*lnR+0.95),0]),1])/(1+exp(-(0.7*log(p[i])-1.1)*(lnR-(-0.07*log(p[i])-2.8))))
end
fduty = 0.97
qwin = 1. - (1.-fduty).^(4./p)
qgeom = 0.005./(p./365.25).^(2/3)
loglog(p,qdet)
loglog(p,qdet.*qwin)
loglog(p,qdet.*qwin.*qgeom)
