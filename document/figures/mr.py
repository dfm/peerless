import numpy as np
import matplotlib.pyplot as pl
from peerless.mr import WolfgangMRRelation, ChenMRRelation

radii = np.linspace(1.0, 12, 100)
for r, c in zip((ChenMRRelation(), WolfgangMRRelation()), "gk"):
    masses = r.predict_mass(radii)
    q = np.percentile(masses, [16, 50, 84], axis=0)
    pl.fill_between(radii, q[0], q[2], color=c, alpha=0.5)
    pl.plot(radii, q[1], color=c, lw=1.5)

pl.yscale("log")
pl.xscale("log")
pl.xlim(radii.min(), radii.max())
pl.xlabel(r"$R/R_\oplus$")
pl.xlabel(r"$M/M_\oplus$")
pl.savefig("mr.png")
