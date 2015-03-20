# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["normalize_inputs", "Model"]

import transit
import numpy as np

# from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import auc, precision_recall_curve

from .settings import HALF_WIDTH


class Model(object):

    def __init__(self, lcs, **kwargs):
        self.lcs = lcs
        self.format_dataset(**kwargs)

    def format_dataset(self, npos=10000, nneg=None, min_period=2e3,
                       max_period=1e4, min_ror=0.03, max_ror=0.3, dt=1.0):
        lcs = self.lcs
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
            ntt = np.random.randint(HALF_WIDTH, len(lcs[nlc])-HALF_WIDTH)
            ror = np.exp(np.random.uniform(np.log(min_ror), np.log(max_ror)))
            pos_pars.append([
                nlc, ntt, lcs[nlc].time[ntt],
                np.random.rand(), np.random.rand(),
                np.exp(np.random.uniform(np.log(min_period),
                                         np.log(max_period))),
                np.random.uniform(-dt, dt),
                ror,
                np.random.uniform(0, 1.0 + ror),
            ])

            # Build the simulator and
            s = simulation_system(*(pos_pars[j][3:]))
            pos_sims[j] = lcs[nlc].flux[ntt+inds] * s.light_curve(t)

        # The negative examples are just random chunks of light curve without
        # any injection.
        neg_sims = np.empty((nneg, len(t)))
        neg_pars = []
        for j in range(len(neg_sims)):
            nlc = np.random.randint(len(lcs))
            ntt = np.random.randint(HALF_WIDTH, len(lcs[nlc])-HALF_WIDTH)
            neg_pars.append([nlc, ntt, lcs[nlc].time[ntt]] + 6*[np.nan])
            neg_sims[j] = lcs[nlc].flux[ntt+inds]

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
        self.X = X[inds]
        self.y = y[inds]
        self.meta = meta[inds]

    def train(self, month, **kwargs):
        # Initialize the model.
        kwargs["n_estimators"] = kwargs.get("n_estimators", 500)
        kwargs["min_samples_leaf"] = kwargs.get("min_samples_leaf", 100)
        clf = ExtraTreesClassifier(**kwargs)

        # Select the training and validation sets.
        r = np.random.rand(len(self.meta)) > 0.5
        m = self.meta["nlc"] != month
        m_train = m & r
        m_valid = m & ~r

        # Train the model.
        clf.fit(self.X[m_train], self.y[m_train])

        # Measure the precision and recall.
        y_valid = clf.predict_proba(self.X[m_valid])[:, 1]
        precision, recall, thresh = precision_recall_curve(self.y[m_valid],
                                                           y_valid)
        print(auc(recall, precision))

        # Test on the held out month.
        two_hw = 2 * HALF_WIDTH
        lc = self.lcs[month]
        inds = np.arange(len(lc)-two_hw)[:, None] + np.arange(two_hw + 1)
        X_test = normalize_inputs(lc.flux[inds])
        y_test = clf.predict_proba(X_test)[:, 1]
        t = lc.time[np.arange(HALF_WIDTH, len(lc)-HALF_WIDTH+1)]

        return precision, recall, thresh, t, y_test


def normalize_inputs(X):
    X /= np.median(X, axis=1)[:, None]
    X[:, :] = np.log(X)
    return X


def simulation_system(q1, q2, period, t0, ror, b):
    s = transit.System(transit.Central(q1=q1, q2=q2))
    s.add_body(transit.Body(period=period, t0=t0, r=ror, b=b))
    return s
