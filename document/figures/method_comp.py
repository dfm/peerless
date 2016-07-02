# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as pl

from peerless.plot_setup import COLORS

names = ["id", "binary", "mass", "radius", "period", "a", "e", "per", "long",
         "asc", "incl", "teq", "age", "method", "year", "updated", "ra", "dec",
         "dist", "s_mass", "s_radius", "s_feh", "s_teff", "s_age"]

oec = pd.read_csv("open_exoplanet_catalogue.txt", skiprows=30, names=names)

for k, _ in oec.groupby("method"):
    print(k, np.sum(oec.method == k))

G = 2945.4625385377644
oec["approx_period"] = 2*np.pi*np.sqrt((215.1*oec.a)**3/G/oec.s_mass)

names = ["transit", "RV", "microlensing"]
hists = {}
m0 = np.isfinite(oec.period)
oec.period[~m0] = oec.approx_period[~m0]
m0 = np.isfinite(oec.period)
bins = np.linspace(np.log10(1.0), np.log10(1e4), 15)
for k in names:
    m = (oec.method == k) & m0
    h, bins = np.histogram(np.log10(oec[m].period), bins)
    hists[k] = h
print(hists["microlensing"])

w = np.diff(10**bins)
bins = 10**bins
bins = bins[:-1]

fig, ax = pl.subplots(1, 1, figsize=(6, 4))
ax.bar(bins, hists["transit"], width=w, color="#6baed6", edgecolor="none",
       label="transit")
ax.bar(bins, hists["RV"], width=w, color="#fd8d3c", edgecolor="none",
       bottom=hists["transit"],
       label="RV")
ax.bar(bins, hists["microlensing"], width=w, color="#74c476", edgecolor="none",
       bottom=hists["transit"] + hists["RV"],
       label="microlensing")
ax.set_xscale("log")
ax.set_xlim(bins[0], bins[-1] + w[-1])

ax.set_xlabel("orbital period [days]")
ax.set_ylabel("number of discoveries")

pl.legend(fontsize=14, loc="upper right")

ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.0f"))

fig.savefig("method_comp.pdf", bbox_inches="tight")
