# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["normalize_inputs", "Model"]

import os
import h5py
import pickle
import transit
import logging
import numpy as np
from scipy.stats import beta

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, precision_recall_curve

from .settings import TEXP, HALF_WIDTH


class Model(object):

    def __init__(self, lcs, **kwargs):
        self.lcs = lcs
        self.models = [None] * len(lcs)
        self.X = None
        self.kwargs = kwargs

    def format_dataset(self, npos=100000, nneg=None,
                       min_period=1e3, max_period=1e4,
                       min_rad=0.03, max_rad=0.3, dt=0.05,
                       smass=1.0, srad=1.0):
        lcs = self.lcs
        if nneg is None:
            nneg = npos

        # Pre-compute the weights for each light curve.
        w = np.array([lc.footprint for lc in lcs])
        w /= np.sum(w)
        self.lc_weights = w

        # The time grid for within a single chunk.
        inds = np.arange(-HALF_WIDTH, HALF_WIDTH+1)
        meta_keys = ["channel", "skygroup", "module", "output", "quarter",
                     "season"]

        # Positive examples.
        pos_sims = np.empty((npos, len(inds)))
        pos_pars = []
        for j in range(len(pos_sims)):
            # Generate the simulation parameters.
            nlc = np.argmax(np.random.multinomial(1, w))
            ntt = np.random.randint(HALF_WIDTH, len(lcs[nlc])-HALF_WIDTH)
            rp = np.exp(np.random.uniform(np.log(min_rad), np.log(max_rad)))
            pos_pars.append([
                nlc, ntt, lcs[nlc].time[ntt],
                np.random.rand(), np.random.rand(),
                np.exp(np.random.uniform(np.log(min_period),
                                         np.log(max_period))),
                np.random.uniform(-dt, dt),
                rp,
                np.random.uniform(0, 1.0 + rp / srad),
                beta.rvs(1.12, 3.09),
                np.random.uniform(-np.pi, np.pi),
            ] + [lcs[nlc].meta[k] for k in meta_keys])

            # Build the simulator and inject the transit signal.
            s = simulation_system(smass, srad,
                                  *(pos_pars[j][3:-len(meta_keys)]))
            t = lcs[nlc].time[ntt+inds]
            t -= np.mean(t)
            pos_sims[j] = lcs[nlc].flux[ntt+inds]*s.light_curve(t, texp=TEXP)

        # The negative examples are just random chunks of light curve without
        # any injection.
        neg_sims = np.empty((nneg, len(t)))
        neg_pars = []
        for j in range(len(neg_sims)):
            nlc = np.argmax(np.random.multinomial(1, w))
            ntt = np.random.randint(HALF_WIDTH, len(lcs[nlc])-HALF_WIDTH)
            neg_pars.append([nlc, ntt, lcs[nlc].time[ntt]] + 8*[np.nan]
                            + [lcs[nlc].meta[k] for k in meta_keys])
            neg_sims[j] = lcs[nlc].flux[ntt+inds]

        # Format the arrays for sklearn.
        X = normalize_inputs(np.concatenate((pos_sims, neg_sims), axis=0))
        y = np.ones(len(X))
        y[len(pos_sims):] = 0

        # Give the metedata a dtype.
        dtype = [("nlc", int), ("ntt", int), ("tt", float),
                 ("q1", float), ("q2", float), ("period", float),
                 ("t0", float), ("rp", float), ("b", float), ("e", float),
                 ("pomega", float)]
        dtype += [(k, int) for k in meta_keys]
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
                    ntrain=None, nvalid=None, **kwargs):
        if not 0 <= month < len(self.lcs):
            raise ValueError("invalid section ID")

        if self.X is None:
            self.format_dataset(**(self.kwargs))

        if self.models[month] is not None and not refit:
            return self.models[month]

        # Generate the two splits of the data.
        s = np.random.get_state()
        m_1 = self._split(month, ntrain, nvalid)
        np.random.set_state(s)
        m_2 = self._split(month, nvalid, ntrain)

        # Initialize the model dictionary.
        self.models[month] = dict(
            section=month,
            splits=[],
        )

        # Loop over the splits and fit the models.
        kwargs["n_estimators"] = kwargs.get("n_estimators", 1000)
        kwargs["min_samples_leaf"] = kwargs.get("min_samples_leaf", 2)
        for m_train, m_valid in (m_1, m_2[::-1]):
            # Initialize the model.
            if cls is None:
                cls = RandomForestClassifier
            clf = cls(**kwargs)

            # Train the model.
            clf.fit(self.X[m_train], self.y[m_train])

            # Predict on the validation set and compute the precision and
            # recall.
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

            self.models[month]["splits"].append(dict(
                classifier=clf,
                precision_recall_curve=prc,
                area_under_the_prc=auc(prc["recall"], prc["precision"]),
                prec_req=prec_req,
                threshold=prc["threshold"][prc["precision"] < prec_req][-1],
                recall=prc["recall"][prc["precision"] < prec_req][-1],
                validation_set=self.meta[m_valid],
                validation_pred=y_valid,
                results=results,
                ntrain=m_train.sum(),
                nvalid=m_valid.sum(),
            ))

        return self.models[month]

    def _split(self, month, ntrain, nvalid):
        # Split the "in quarter" sections into the training/validation sets.
        lc = self.lcs[month]
        quarters = np.array([l.meta["quarter"] for l in self.lcs])
        seasons = np.array([l.meta["season"] for l in self.lcs])

        # Split the "in quarter" sections.
        m = np.ones_like(quarters, dtype=bool)
        m[month] = False
        in_quarter = quarters == lc.meta["quarter"]
        inds = np.arange(len(quarters))[in_quarter & m]
        np.random.shuffle(inds)
        w = self.lc_weights[inds]
        w /= np.sum(w)
        cs = np.cumsum(w) <= 0.5
        train_lcs = inds[cs]
        valid_lcs = inds[~cs]

        # Split the "in season" sections.
        in_season = seasons == lc.meta["season"]
        inds = np.arange(len(quarters))[in_season & ~in_quarter & m]
        np.random.shuffle(inds)
        w = self.lc_weights[inds]
        w /= np.sum(w)
        cs = np.cumsum(w) <= 0.5
        train_lcs = np.append(train_lcs, inds[cs])
        valid_lcs = np.append(valid_lcs, inds[~cs])

        # Select the training and validation samples.
        train_lcs, valid_lcs = set(train_lcs), set(valid_lcs)
        m_train = np.array([row["nlc"] in train_lcs for row in self.meta])
        m_valid = np.array([row["nlc"] in valid_lcs for row in self.meta])

        # Select the extra samples that might be needed if there aren't enough
        # in season and in quarter samples.
        inds = np.arange(len(quarters))[quarters != lc.meta["quarter"]]
        np.random.shuffle(inds)
        w = self.lc_weights[inds]
        w /= np.sum(w)
        cs = np.cumsum(w) <= 0.5
        train_lcs, valid_lcs = set(inds[cs]), set(inds[~cs])
        x_train = np.array([row["nlc"] in train_lcs for row in self.meta])
        x_valid = np.array([row["nlc"] in valid_lcs for row in self.meta])

        # Only select the requested number of training/validation examples.
        if ntrain is None:
            # Add in the extra samples.
            m_train |= x_train
        else:
            m_train[np.cumsum(m_train) > ntrain] = False
            if m_train.sum() != ntrain:
                x_train[np.cumsum(x_train) > ntrain - m_train.sum()] = False
                m_train |= x_train
            if m_train.sum() != ntrain:
                logging.warn("Not enough training examples ({0})"
                             .format(m_train.sum()))
        if nvalid is None:
            # Add in the extra samples.
            m_valid |= x_valid
        else:
            m_valid[np.cumsum(m_valid) > nvalid] = False
            if m_valid.sum() != ntrain:
                x_valid[np.cumsum(x_valid) > nvalid - m_valid.sum()] = False
                m_valid |= x_valid
            if m_valid.sum() != nvalid:
                logging.warn("Not enough validation examples ({0})"
                             .format(m_valid.sum()))

        return m_train, m_valid

    def to_hdf(self, fn):
        fn = os.path.abspath(fn)
        try:
            os.makedirs(os.path.dirname(fn))
        except os.error:
            pass

        with h5py.File(fn, "w") as f:
            for results in self.models:
                if results is None:
                    continue

                # Create the data group and save the attributes.
                sect = results["section"]
                g0 = f.create_group("section_{0:03d}".format(sect))
                g0.attrs["section"] = sect

                # Loop over the splits and save those.
                for i, res in enumerate(results["splits"]):
                    g = g0.create_group("split_{0:d}".format(i))
                    for k in ["prec_req", "threshold", "recall",
                            "area_under_the_prc"]:
                        g.attrs[k] = res[k]
                    g.attrs["classifier"] = pickle.dumps(res["classifier"])

                    # Save the datasets.
                    for k in ["precision_recall_curve", "validation_set",
                              "validation_pred", "results"]:
                        g.create_dataset(k, data=res[k], compression="gzip")

    @classmethod
    def from_hdf(cls, fn, lcs, **kwargs):
        self = cls(lcs, **kwargs)
        with h5py.File(fn, "r") as f:
            for k in f:
                g0 = f[k]
                d0 = dict(section=g0.attrs["section"], splits=[])
                for k in g0:
                    g = g0[k]
                    d = dict(classifier = pickle.loads(g.attrs["classifier"]))
                    for k in ["prec_req", "threshold", "recall",
                              "area_under_the_prc"]:
                        d[k] = g.attrs[k]

                    for k in ["precision_recall_curve", "validation_set",
                            "validation_pred", "results"]:
                        d[k] = g[k][...]
                    d0["splits"].append(d)
                self.models[g0.attrs["section"]] = d0
        return self


def normalize_inputs(X):
    X /= np.median(X, axis=1)[:, None]
    X[:, :] = np.log(X)
    return X


def simulation_system(smass, srad, q1, q2, period, t0, rp, b, e, pomega):
    s = transit.System(transit.Central(mass=smass, radius=srad, q1=q1, q2=q2))
    s.add_body(transit.Body(period=period, t0=t0, r=rp, b=b, e=e,
                            pomega=pomega))
    return s
