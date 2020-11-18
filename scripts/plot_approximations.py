"""
Evaluate the accuracy of Newton iterations for 1/x, 1/sqrt(x), x^{-1/8},
and sgn(x).
"""

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import matplotlib.pyplot as plt


# Newton iterations for x -> 1/x
def reciprocal(x):
    y = 1
    for k in range(30):
        y = y * (2 - x * y)
    return y


# Newton iterations for x -> 1/sqrt(x)
def invsqrt(x):
    y = 1
    for k in range(26):
        y = y * (3 - x * y**2) / 2
    return y


# Newton iterations for x -> x^{-1/8}
def inv8root(x):
    y = 1
    for k in range(24):
        y = y * (9 - x * y**8) / 8
    return y


# Newton iterations for x -> sgn(x), followed by sgn(x) -> x * sgn(x) = |x|
def absval(x):
    y = x / 1e5
    for k in range(60):
        y = y * (3 - y**2) / 2
    return y * x


# Set the number of samples for the abscissa.
n = 20000

# Plot the relative accuracy for reciprocal.
x = [1.9999999 * math.exp((-17 * k) / n) for k in range(n)]
y = [(1 / x[k] - reciprocal(x[k])) * x[k] for k in range(n)]
plt.figure(figsize=(6, 4))
plt.xlabel(r'$x$')
plt.ylabel(r'$\frac{1/x - \widetilde{1/x}}{1/x}$', rotation=0, labelpad=15)
plt.xscale('log')
plt.minorticks_off()
plt.plot(x, y, 'k')
plt.savefig('results/reciprocal.pdf', bbox_inches='tight')

# Plot the relative accuracy for invsqrt.
x = [2.99 * math.exp(-18.125 * k / n) for k in range(n)]
y = [(1 / math.sqrt(x[k]) - invsqrt(x[k])) * math.sqrt(x[k]) for k in range(n)]
plt.figure(figsize=(6, 4))
plt.xlabel(r'$x$')
plt.ylabel(r'$\frac{1/\sqrt{x} - \widetilde{1/\sqrt{x}}}{1/\sqrt{x}}$',
           rotation=0, labelpad=20)
plt.xscale('log')
plt.minorticks_off()
plt.plot(x, y, 'k')
plt.savefig('results/invsqrt.pdf', bbox_inches='tight')

# Plot the relative accuracy for inv8root.
x = [8.25 * math.exp(-20 * k / n) for k in range(n)]
y = [(1 / math.sqrt(math.sqrt(math.sqrt(x[k]))) - inv8root(x[k]))
     * math.sqrt(math.sqrt(math.sqrt(x[k]))) for k in range(n)]
plt.figure(figsize=(6, 4))
plt.xlabel(r'$x$')
plt.ylabel(r'$\frac{x^{-1/8} - \widetilde{x^{-1/8}}}{x^{-1/8}}$',
           rotation=0, labelpad=20)
plt.xscale('log')
plt.minorticks_off()
plt.plot(x, y, 'k')
plt.savefig('results/inv8root.pdf', bbox_inches='tight')

# Plot the absolute accuracy for absval.
x1 = [1e5 * 1.7 * math.exp(-51 * k / n) for k in range(n)]
x2 = [-x1[k] for k in range(n)]
y1 = [abs(x1[k]) - absval(x1[k]) for k in range(n)]
y2 = [abs(x2[k]) - absval(x2[k]) for k in range(n)]
plt.figure(figsize=(6, 4))
plt.xlabel(r'$x$')
plt.ylabel(r'$|x| - \widetilde{|x|}$', rotation=0, labelpad=30)
plt.xscale('log')
plt.minorticks_off()
plt.plot(x1, y1, 'gray', linewidth=5)
plt.plot(x1, y2, 'k:', linewidth=2)
plt.savefig('results/absval.pdf', bbox_inches='tight')
