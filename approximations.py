#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np


def chebyshev(n, x):
    """
    evaluates the Chebyshev polynomials of odd degrees at x

    Parameters
    ----------
    n
        highest degree of Cheybshev polynomials to evaluate;
        the ndarray returned has n entries, from 0 to n-1
    x
        point at which to evaluate the Chebyshev polynomials

    Returns
    -------
    list
        values of the odd-degree Chebyshev polynomials at x

    N.B. The input n must be even and at least 6.
    """
    assert n % 2 == 0, "chebyshev degree should be even"
    assert n >= 6, "chebyshev degree should be > 6"
    t = []
    t.append(x.clone())
    y = 4 * x.square() - 2
    z = y - 1
    t.append(z.mul(x))
    for k in range(2, n // 2):
        t.append(y * t[k - 1] - t[k - 2])
    return t


def chebseries(f, w, n):
    """
    calculates the Chebyshev coefficients up to degree n-1 for f on [-w, w]

    Parameters
    ----------
    f
        function whose Chebyshev coefficients will be calculated
    w
        upper limit of the domain [-w, w] on which f will be approximated
    n
        lowest degree of polynomials not included in the approximation

    Returns
    -------
    ndarray
        coefficients in the Chebyshev expansion
    """
    x = w * np.array([math.cos((k + 0.5) * math.pi / n) for k in range(n)])
    y = np.array([f(x[k]) for k in range(n)])
    ks = np.arange(n)[np.newaxis, :]
    js = np.arange(n)[:, np.newaxis]
    c = (2 / n) * np.sum(y * np.cos(js * (ks + 0.5) * np.pi / n), axis=1)
    c[0] /= 2
    return c
