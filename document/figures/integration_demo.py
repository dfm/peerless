#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
A test for how to do the integral

I = \int_{\ln x_1}^{\ln x_2} f(\ln x) x^{-5/3} d\ln x
  = \int_{x_1}^{x_2} f(\ln x) x^{-5/3} x^{-1} dx

if f(\ln x) = f0:

I = f0 \, (x_2^{-5/3} - x_1^{-5/3}) / (-5/3)

if f(\ln x) = x^b:

I = (x_2^{-5/3+b} - x_1^{-5/3+b}) / (-5/3+b)

"""

import numpy as np


def sample_power_law(n, mn, mx, size=None):
    u = np.random.uniform(size=size)
    np1 = n + 1.0
    a = mx**np1
    b = mn**np1
    return (
        (u*a + (1-u)*b) ** (1. / np1),
        (a - b) / np1
    )


power = -5/3

f0 = 0.1
mn, mx = 0.5, 10.0
samples, norm = sample_power_law(power-1.0, mn, mx, 2000000)
print(np.mean(f0 * np.ones_like(samples)) * norm, f0 * norm)

beta = -0.5
beta_plus = beta + power
print(np.mean(samples**beta)*norm, (mx**beta_plus-mn**beta_plus)/beta_plus)
