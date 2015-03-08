# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = []

import kplr
import transit
import numpy as np

from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import auc, precision_recall_curve

HALF_WIDTH = 60


def load_light_curves(kicid):
    client = kplr.API()

    lcs = []
    for lc in client.star(kicid).get_light_curves(short_cadence=False):
        # Load the data.
        data = lc.read()
        x = data["TIME"]
        y = data["PDCSAP_FLUX"]

        # Split into months.
        m = np.isfinite(x)
        gi = np.arange(len(x))[m]
        bi = np.arange(len(x))[~m]
        if len(bi):
            bi = bi[(bi > gi[0]) & (bi < gi[-1])]
            d = np.diff(bi)
            chunks = [slice(gi[0], bi[0])]
            for a, b in zip(bi[:-1][d > 1], bi[1:][d > 1]):
                chunks.append(slice(a+1, b-1))
            chunks.append(slice(bi[-1]+1, gi[-1]))
        else:
            chunks = [slice(gi[0], gi[-1])]

        # Interpolate missing data.
        for c in chunks:
            x0, y0 = x[c], y[c]
            m = np.isfinite(y0)
            if not np.any(m):
                continue
            y0[~m] = np.interp(x0[~m], x0[m], y0[m])
            lcs.append([x0, y0 / np.median(y0)])

    # Only retain chunks that are long enough (wrt the window half width).
    return [lc for lc in lcs if len(lc[0]) > 2 * HALF_WIDTH]


def simulation_system(q1, q2, period, t0, ror, b):
    s = transit.System(transit.Central(q1=q1, q2=q2))
    s.add_body(transit.Body(period=period, t0=t0, r=ror, b=b))
    return s


def normalize_inputs(X):
    X /= np.median(X, axis=1)[:, None]
    X[:, :] = np.log(X)
    return X


def generate_injections(lcs, npos=10000, nneg=None,
                        min_period=2e3, max_period=1e4,
                        min_ror=0.03, max_ror=0.3):
    if nneg is None:
        nneg = npos

    # The time grid for within a single chunk.
    inds = np.arange(-HALF_WIDTH, HALF_WIDTH+1)
    t = 0.5 / 24. * inds

    # Positive examples.
    pos_sims = np.empty((npos, len(t)))
    pos_pars = []
    for j in range(len(pos_sims)):
        # Generate the simulation parameters.
        nlc = np.random.randint(len(lcs))
        ntt = np.random.randint(HALF_WIDTH, len(lcs[nlc][0])-HALF_WIDTH)
        pos_pars.append([
            nlc, ntt, lcs[nlc][0][ntt],
            np.random.rand(), np.random.rand(),
            np.exp(np.random.uniform(np.log(min_period), np.log(max_period))),
            np.random.uniform(-0.5, 0.5),
            np.exp(np.random.uniform(np.log(min_ror), np.log(max_ror))),
            np.random.rand(),
        ])

        # Build the simulator and
        s = simulation_system(*(pos_pars[j][3:]))
        pos_sims[j] = lcs[nlc][1][ntt+inds] * s.light_curve(t)

    # The negative examples are just random chunks of light curve without any
    # injection.
    neg_sims = np.empty((nneg, len(t)))
    neg_pars = []
    for j in range(len(neg_sims)):
        nlc = np.random.randint(len(lcs))
        ntt = np.random.randint(HALF_WIDTH, len(lcs[nlc][0])-HALF_WIDTH)
        neg_pars.append([nlc, ntt, lcs[nlc][0][ntt]] + 6*[np.nan])
        neg_sims[j] = lcs[nlc][1][ntt+inds]

    # Format the arrays for sklearn.
    X = normalize_inputs(np.concatenate((pos_sims, neg_sims), axis=0))
    y = np.ones(len(X))
    y[len(pos_sims):] = 0

    # Give the metedata a dtype.
    dtype = [("nlc", int), ("ntt", int), ("tt", float),
             ("q1", float), ("q2", float), ("period", float),
             ("t0", float), ("ror", float), ("b", float)]
    meta = np.array(map(tuple, pos_pars + neg_pars), dtype=dtype)

    # Shuffle the order.
    inds = np.arange(len(X))
    np.random.shuffle(inds)
    X, y, meta = X[inds], y[inds], meta[inds]

    return X, y, meta


def _pr_scorer(model, X, y):
    prec, rec, _ = precision_recall_curve(y, model.predict_proba(X)[:, 1])
    return auc(rec, prec)


def cross_validate(X, y):
    parameters = dict(
        n_estimators=[500, 1500, 2000],
        min_samples_leaf=[3, 5],
    )

    clf = ExtraTreesClassifier()
    gs = GridSearchCV(clf, parameters, scoring=_pr_scorer, n_jobs=-1)

    return gs.fit(X, y)
