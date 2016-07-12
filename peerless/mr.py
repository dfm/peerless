# -*- coding: utf-8 -*-

from __future__ import division, print_function

import logging
import numpy as np

from .catalogs import LocalCatalog, singleton

__all__ = ["WolfgangMRRelation"]


class WolfgangMRRelation(LocalCatalog):
    filename = "wolfgang.csv"

    def predict_mass(self, radius_samples, num_hyper=None, maxiter=500):
        radius_samples = np.atleast_1d(radius_samples)
        shape = radius_samples.shape
        flat_radii = radius_samples.flatten()
        params = self.df

        inds = np.ones(len(params), dtype=bool)
        if num_hyper is not None:
            inds = np.random.randint(len(params), size=num_hyper)

        # Grab the parameter samples.
        lnc = np.array(params["normconst"])[inds]
        gamma = np.array(params["powindex"])[inds]
        sigm = np.sqrt(np.array(params["varMR"]))[inds]

        # Use Wolfgang+ (2016) Equation (2).
        mu = lnc[:, None] + gamma[:, None] * np.log(flat_radii)[None, :]
        std = sigm[:, None] + np.zeros_like(mu)
        mass = np.exp(mu) + std * np.random.randn(*(mu.shape))

        # Iterate until none of the masses are < 0.0.
        for i in range(maxiter):
            m = mass < 0.0
            if not np.any(m):
                break
            mass[m] = np.exp(mu[m]) + std[m] * np.random.randn(m.sum())
        if i == maxiter - 1:
            logging.warn("Some masses still incorrect after 'maxiter'")

        # For R > 9 R_E, use a log-normal distribution.
        mask = flat_radii > 9.0
        lnm = 0.04590711 + 0.3919828*np.random.randn(len(mass), mask.sum())
        mass[:, mask] = np.exp(lnm + np.log(317.828))

        # Reshape the samples into the correct (num_hyper, ...) shape.
        final_shape = [len(lnc)] + list(shape)
        return mass.reshape(final_shape)


WolfgangMRRelation = singleton(WolfgangMRRelation)
