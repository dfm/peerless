# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["normalize_inputs", "Model"]

import os
import h5py
import pickle
import transit
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, precision_recall_curve

from .settings import TEXP, HALF_WIDTH


class Model(object):

    def __init__(self, lcs, **kwargs):
        self.lcs = lcs
        self.models = [None] * len(lcs)
        self.format_dataset(**kwargs)

    def format_dataset(self, npos=10000, nneg=None,
                       min_period=500, max_period=1e4,
                       min_ror=0.03, max_ror=0.3, dt=0.1):
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

            # Build the simulator and inject the transit signal.
            s = simulation_system(*(pos_pars[j][3:]))
            pos_sims[j] = lcs[nlc].flux[ntt+inds]*s.light_curve(t, texp=TEXP)

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

    def fit_all(self, **kwargs):
        return [self.fit_section(i, **kwargs) for i in range(len(self.lcs))]

    def fit_section(self, month, refit=False, cls=None, prec_req=0.9999,
                    **kwargs):
        if not 0 <= month < len(self.lcs):
            raise ValueError("invalid section ID")

        if self.models[month] is not None and not refit:
            return self.models[month]

        # Initialize the model.
        kwargs["n_estimators"] = kwargs.get("n_estimators", 500)
        kwargs["min_samples_leaf"] = kwargs.get("min_samples_leaf", 1)
        if cls is None:
            cls = RandomForestClassifier
        clf = cls(**kwargs)

        # Select the training and validation sets.
        r = np.random.rand(len(self.meta)) > 0.5
        m = self.meta["nlc"] != month
        m_train = m & r
        m_valid = m & ~r

        # Train the model.
        clf.fit(self.X[m_train], self.y[m_train])

        # Predict on the validation set and compute the precision and recall.
        y_valid = clf.predict_proba(self.X[m_valid])[:, 1]
        prc = precision_recall_curve(self.y[m_valid], y_valid)
        prc = np.array(zip(prc[0], prc[1], np.append(prc[2], 1.0)),
                       dtype=[("precision", float), ("recall", float),
                              ("threshold", float)])

        # Test on the held out month.
        two_hw = 2 * HALF_WIDTH
        lc = self.lcs[month]
        inds = np.arange(len(lc)-two_hw)[:, None] + np.arange(two_hw + 1)
        X_test = normalize_inputs(lc.flux[inds])
        y_test = clf.predict_proba(X_test)[:, 1]
        t = lc.time[np.arange(HALF_WIDTH, len(lc)-HALF_WIDTH+1)]
        results = np.array(zip(t, y_test), dtype=[("time", float),
                                                  ("predict_prob", float)])

        self.models[month] = dict(
            section=month,
            classifier=clf,
            precision_recall_curve=prc,
            area_under_the_prc=auc(prc["recall"], prc["precision"]),
            prec_req=prec_req,
            threshold=prc["threshold"][prc["precision"] < prec_req][-1],
            recall=prc["recall"][prc["precision"] < prec_req][-1],
            validation_set=self.meta[m_valid],
            validation_pred=y_valid,
            results=results,
        )

        return self.models[month]

    def to_hdf(self, fn):
        fn = os.path.abspath(fn)
        try:
            os.makedirs(os.path.dirname(fn))
        except os.error:
            pass

        with h5py.File(fn, "w") as f:
            for res in self.models:
                if res is None:
                    continue

                # Create the data group and save the attributes.
                g = f.create_group("section_{0:03d}".format(res["section"]))
                for k in ["section", "prec_req", "threshold", "recall",
                          "area_under_the_prc"]:
                    g.attrs[k] = res[k]
                g.attrs["classifier"] = pickle.dumps(res["classifier"])

                # Save the datasets.
                for k in ["precision_recall_curve", "validation_set",
                          "validation_pred", "results"]:
                    g.create_dataset(k, data=res[k])

    @classmethod
    def from_hdf(cls, fn, lcs, **kwargs):
        self = cls(lcs, **kwargs)
        with h5py.File(fn, "r") as f:
            for k in f:
                g = f[k]
                d = dict(classifier = pickle.loads(g.attrs["classifier"]))
                for k in ["section", "prec_req", "threshold", "recall",
                          "area_under_the_prc"]:
                    d[k] = g.attrs[k]

                for k in ["precision_recall_curve", "validation_set",
                          "validation_pred", "results"]:
                    d[k] = g[k][...]
                self.models[g.attrs["section"]] = d
        return self


def normalize_inputs(X):
    X /= np.median(X, axis=1)[:, None]
    X[:, :] = np.log(X)
    return X


def simulation_system(q1, q2, period, t0, ror, b):
    s = transit.System(transit.Central(q1=q1, q2=q2))
    s.add_body(transit.Body(period=period, t0=t0, r=ror, b=b))
    return s
