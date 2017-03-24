using CGS
# Period estimates for Dan's sample assuming e=0:
ncand = 16
KIC = ["3218908","3239945","4754460","6551440","8410697","8426957","8505215","8738735","8800954","9306307","10187159","10287723","10321319","10602068","10842718","11709124"]
b = [0.,0.207,0.893,0.75,0.,0.889,0.,0.,0.,0.64,0.,0.8,0.7,0.6027,0.,0.55]
rho = [2.716,2.963,0.8725,1.053,1.201,1.145,3.139,1.388,2.502,1.383,1.553,2.64,1.414,1.762,1.177,1.377]
srho1 = [1.432,0.4687,0.476,0.6332,0.6885,0.7186,0.6771,0.5794,0.4743,0.9142,1.3,0.2721,0.8579,1.09,0.6849,0.5512]
srho2 = [0.3974,0.2218,0.7728,0.5275,0.7425,0.4541,0.2093,0.1768,0.3012,1.01,0.6905,0.4805,0.7137,0.3954,0.5948,0.465]
tdur = [21.45,16.202,15.92,10.85,19.77,39.4,20.06,27.44,15.76,8.499,11.81,9.49,16.84,12.804,35.92,17.75]
rad = [0.514,0.876,0.67,0.282,0.698,1.04,0.277,0.355,0.386,1.22,0.43,0.266,0.163,2.0,0.74,0.83]*RJUPITER/REARTH
srad = [0.092,0.039,0.16,0.093,0.107,0.30,0.017,0.045,0.025,0.49,0.21,0.027,0.046,0.66,0.16,0.12]
#rho  = 3/pi^2*(period*24.*3600.)/GRAV/(tdur*3600.)^3*(1.-b^2)^1.5
fac = pi^2/3/24./3600.*3600.^3*GRAV
p = fac.*rho.*tdur.^3./(1-b.^2).^1.5
p1 = p.*(1.-srho1./rho)
p2 = p.*(1.+srho2./rho)
p[2]=2.9328721*365.25
p1[2]=(2.9328721-2.6e-6)*365.25
p2[2]=(2.9328721+2.6e-6)*365.25
p[5]=2.8688097*365.25
p1[5]=(2.8688097-5.3e-6)*365.25
p2[5]=(2.8688097+5.4e-6)*365.25
p[9]=1.9279957*365.25
p1[9]=(1.9279957-9.1e-6)*365.25
p2[9]=(1.9279957+9.2e-6)*365.25
using PyPlot
clf()
for i=1:ncand
  loglog([p1[i],p2[i]],[rad[i],rad[i]])
  loglog([p[i],p[i]],[rad[i]-srad[i],rad[i]+srad[i]])
#  println(KIC[i]," ",p1[i]/365.25," ",p2[i]/365.25," ",rad[i])
  println(KIC[i]," ",p[i]/365.25," ",rad[i])
end
#plot([780,780],[1,100])
#plot([4*365,4*365],[1,100])
plot([600,600],[1,100])
plot([1000,1000],[1,100])

# Notes on individual systems:
# KIC 3218908 - high probability of being a planet, and inner planets agree in duration/period
#   diagram.  The parameters look reasonable, so this is a likely candidate. [High quality]
#
# KIC 3239945 - high FAP (27%); but it has multiple transits, and is consistent with inner planets.
#   The parameters look well determined.  This is a high quality candidate. [High quality]
#
# KIC 4754460 - Likely eclipse, high b, so probably FP. [False alarm]
#
# KIC 6551440 - Has a most likely period that is quite short.  High b, broad eccentricity. 
#  FAP is modest 3%.  Perhaps stellar parameters are off? [Stellar params?]
#
# KIC 8410697 - Multiple transits, so well determined period.  Parameters look well constrained.
#  This is a high quality candidate. [High quality]
#
# KIC 8426957 - FAP is high 20%; high b.  Period is *very* long, which seems unlikely.  Perhaps stellar
#  parameters are off. [Stellar params?]
#
# KIC 8505215 - There is a short-period candidate 99.02 which is labeled as a false-positive (may be a blend).  
#  DV summary shows an offset during transit. Dan's candidate looks good, though. [Blend?]
#
# KIC 8738735 - Stellar density inferred from inner two planets is inconsistent with Huber's density.
#  Otherwise the parameters look reasonable. [Stellar params?]
#
# KIC 8800954 - Multiple transits, parameters look good. [High quality]
#
# KIC 9306307 - High impact parameter (multi-modal), high eccentricity. Inferred period for e=0
#  is quite short.  Perhaps stellar parameters are off? [Stellar params?]
#
# KIC 10187159 - Density inferred from inner planet is much higher than Huber's.  Parameters of fit look
#  fine, but peaked at very short period. FAP is 9%. [Stellar params?]
#
# KIC 10287723 - High impact parameter, high eccentricity.  Period peaks at ~few years.  Low FAP (5%).
#  [High quality]
# 
# KIC 10321319 - Low FAP; parameters look good. [High quality]
#
# KIC 10602068 - High FAP (~100%). [False alarm]
#
# KIC 10842718 - 10% FAP, but parameters look good. [High quality]
#
# KIC 11709124 - Parameters look good. [High quality]
