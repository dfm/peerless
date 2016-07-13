# -*- coding: utf-8 -*-

from __future__ import division, print_function

import h5py
import logging
import numpy as np

from .catalogs import Catalog, LocalCatalog, singleton

__all__ = ["WolfgangMRRelation"]


class WolfgangMRRelation(LocalCatalog):
    filename = "wolfgang.csv"

    def predict_mass(self, radius_samples, num_mass=None, maxiter=500):
        radius_samples = np.atleast_1d(radius_samples)
        shape = radius_samples.shape
        flat_radii = radius_samples.flatten()
        params = self.df

        inds = np.ones(len(params), dtype=bool)
        if num_mass is not None:
            inds = np.random.randint(len(params), size=num_mass)

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

        # Reshape the samples into the correct (num_mass, ...) shape.
        final_shape = [len(lnc)] + list(shape)
        return mass.reshape(final_shape)


class ChenMRRelation(Catalog):
    url = ("https://github.com/chenjj2/forecaster/blob/master/"
           "fitting_parameters.h5?raw=true")
    name = "chen"
    _grid = None

    def _save_fetched_file(self, file_handle):
        with open(self.filename, "wb") as f:
            f.write(file_handle.read())
        with h5py.File(self.filename, "r") as f:
            log_r, log_m, grid = self.make_grid(f["hyper_posterior"][...])
        with h5py.File(self.filename, "w") as f:
            f.create_dataset("log10_radius_bins", data=log_r)
            f.create_dataset("log10_mass_bins", data=log_m)
            f.create_dataset("cumulative_probability", data=grid)

    def make_grid(self, samples):
        """
        Convert the model from Chen & Kipping to ``p(log(M) | log(R))``

        The idea here is their model gives ``p(log(R) | log(M)) =
        N(mu(M), std(M))`` and, by the chain rule, we can compute
        ``p(log(M) | log(R)) = p(log(R) | log(M)) * p(log(M)) / p(log(R))``.
        If we assume a flat prior in ``log(M)``, we can numerically compute
        the required probability distribution on a grid.

        """
        log_mass = np.linspace(np.log10(1e-4), np.log10(1e6), 800)
        log_radius = np.linspace(np.log10(0.1), np.log10(100.0), 901)

        # Get the parameters into the correct shape.
        npop = (samples.shape[-1] - 1) // 3 + 1
        slope = samples[:, 1:1+npop]
        sigma = samples[:, 1+npop:1+2*npop]
        split = samples[:, 1+2*npop:3*npop]
        const = np.empty_like(slope)
        const[:, 0] = samples[:, 0]
        for i in range(1, npop):
            delta = slope[:, i-1] - slope[:, i]
            const[:, i] = const[:, i-1] + split[:, i-1] * delta

        grid = -np.inf + np.zeros((len(log_radius), len(log_mass)))
        dm = log_mass[1] - log_mass[0]
        print("Computing grid -- this might take a minute...")
        for i in np.random.randint(0, len(samples), 2048):
            log_mass_bins = np.concatenate(([log_mass[0]-dm],
                                            split[i],
                                            [log_mass[-1]+dm]))
            inds = np.digitize(log_mass, log_mass_bins) - 1
            mu = log_mass * slope[i, inds] + const[i, inds]
            std = sigma[i, inds]

            log_pred = -0.5 * ((log_radius[:, None]-mu)/std)**2
            log_pred -= np.log(std)
            grid = np.logaddexp(grid, log_pred)

        # Normalize the grid as p(log M | log R)
        grid = np.cumsum(np.exp(grid), axis=1)
        grid /= grid[:, -1][:, None]

        dr = log_radius[1] - log_radius[0]
        log_mass_bins = np.append(log_mass[0]-0.5*dr, log_mass+0.5*dr)
        log_radius_bins = np.append(log_radius[0]-0.5*dr, log_radius+0.5*dr)
        return log_radius_bins, log_mass_bins, grid

    def open(self):
        return h5py.File(self.filename, "r")

    @property
    def grid(self):
        if self._grid is None:
            self.fetch()
            with h5py.File(self.filename, "r") as f:
                self._grid = dict(
                    log10_radius_bins=f["log10_radius_bins"][...],
                    log10_mass_bins=f["log10_mass_bins"][...],
                    cumulative_probability=f["cumulative_probability"][...],
                )
        return self._grid

    def predict_mass(self, radius_samples, num_mass=10000):
        radius_samples = np.atleast_1d(radius_samples)
        shape = radius_samples.shape
        flat_log_r = np.log10(radius_samples.flatten())

        g = self.grid
        log_r = g["log10_radius_bins"]
        log_m = g["log10_mass_bins"]
        grid = g["cumulative_probability"]

        r_inds = np.digitize(flat_log_r, log_r) - 1
        if np.any(r_inds < 0) or np.any(r_inds >= len(log_r) - 1):
            logging.warn("Radii outside of grid")
            r_inds[r_inds < 0] = 0
            r_inds[r_inds >= len(log_r) - 1] = len(log_r) - 2

        # Sample the masses.
        cp = grid[r_inds]
        mass = np.empty((num_mass, len(cp)))
        dm = log_m[1] - log_m[0]
        for i, c in enumerate(cp):
            u = np.random.rand(num_mass)
            j = np.digitize(u, c) - 1
            mass[:, i] = 10**(log_m[j] + dm * np.random.rand(num_mass))

        # Reshape the samples into the correct (num_mass, ...) shape.
        final_shape = [num_mass] + list(shape)
        return mass.reshape(final_shape)


WolfgangMRRelation = singleton(WolfgangMRRelation)
ChenMRRelation = singleton(ChenMRRelation)
