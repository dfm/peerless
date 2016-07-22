using PyPlot
using CGS
nperiod = 100
p = logspace(log10(2),log10(5000/365.25),nperiod)
lnR = log(4*REARTH/RJUPITER)
qdet = zeros(nperiod)
for i=1:nperiod
  qdet[i] = minimum([maximum([(-0.13*log(p[i])+0.95)*(-0.20*lnR+0.90),0]),1])/(1+exp(-(0.7*log(p[i])+3.06)*(lnR-(-0.07*log(p[i])-0.91))))
end
fduty = 0.97
qwin = 1. - (1.-fduty).^(4./p)
qgeom = 0.005./p.^(2/3)
#loglog(p,qdet)
#loglog(p,qdet.*qwin)
semilogx(p*365.25,qdet.*qwin.*qgeom.*1e4)
plot([200,200,340,340,590,590],[0,15,15,12,12,0])
plot([590,590,1000,1000,1700,1700,3000,3000,5000,5000],[0,1,1,6,6,5,5,3,3,0])
