# -*- coding: utf-8 -*-
"""Barabanov norms.

Created on Sat Sep 21 12:37:46 2019.
Last updated on Sat Sep 17 13:17:44 2022 +0300

@author: Victor Kozyakin
"""
import math
import platform
import time
from importlib.metadata import version

import numpy as np
import shapely
from matplotlib import pyplot
from shapely.geometry import LineString, MultiPoint


def polygonal_norm(_x, _y, _h):
    """Calculate the norm specified by a polygonal unit ball.

    Args:
        _x (real): x-coordinate of vector
        _y (real): y-coordinate of vector
        _h (MultiPoint): polygonal norm unit ball

    Returns:
        real: vector's norm
    """
    _hb = _h.bounds
    _scale = 0.5 * math.sqrt(((_hb[2] - _hb[0])**2 + (_hb[3] - _hb[1])**2) /
                             (_x**2 + _y**2))
    _ll = LineString([(0, 0), (_scale*_x, _scale*_y)])
    _h_int = _ll.intersection(_h).coords
    return math.sqrt((_x**2 + _y**2) / (_h_int[1][0]**2 + _h_int[1][1]**2))


def min_max_norms_quotent(_g, _h):
    """Calculate the min/max of the quotient g-norm/h-norm.

    Args:
        _g (MultiPoint): polygonal norm unit ball
        _h (MultiPoint): polygonal norm unit ball

    Returns:
        2x0-array: mimimum and maximum of g-norm/h-norm
    """
    _pg = _g.boundary.coords
    _dimg = len(_pg) - 1
    _sg = [1 / polygonal_norm(_pg[i][0], _pg[i][1], _h) for i in range(_dimg)]
    _ph = _h.boundary.coords
    _dimh = len(_ph) - 1
    _sh = [polygonal_norm(_ph[i][0], _ph[i][1], _g) for i in range(_dimh)]
    _sgh = _sg + _sh
    return (min(_sgh), max(_sgh))

# Initialization


t0 = time.time()
T_COMP = 0.
NITER = 0.

TOL = 0.0000001
A0 = np.asarray([[3/5, -4/5],  [4/5, 3/5]])
A1 = np.asarray([[3/5, -4/6], [24/25, 3/5]])

# Computation

if ((np.linalg.det(A0) == 0) or (np.linalg.det(A1) == 0)):
    raise SystemExit("Set of matrices is degenerate. End of work!")

invA0 = np.linalg.inv(A0)
invA1 = np.linalg.inv(A1)
invA0T = np.transpose(invA0)
invA1T = np.transpose(invA1)

p0 = np.asarray([[1, -1], [1, 1]])
p0 = np.concatenate((p0, -p0), axis=0)
p0 = MultiPoint(p0)
h0 = p0.convex_hull

scale0 = 1 / max(h0.bounds[2], h0.bounds[3])
h0 = shapely.affinity.scale(h0, xfact=scale0, yfact=scale0)

t00 = time.time()

print("\n  #   rho_min   gamma   rho_max  Num_edges\n")

# Computations

while True:
    t1 = time.time()

    p0 = np.asarray(MultiPoint(h0.boundary.coords))

    p1 = np.matmul(p0, invA0T)
    p1 = MultiPoint(p1)
    h1 = p1.convex_hull

    p2 = np.matmul(p0, invA1T)
    p2 = MultiPoint(p2)
    h2 = p2.convex_hull

    h12 = h1.intersection(h2)
    p12 = h12.boundary.coords
    p12 = MultiPoint(p12)

    rho_minmax = min_max_norms_quotent(h12, h0)
    rho_max = rho_minmax[1]
    rho_min = rho_minmax[0]

    gamma = (rho_max + rho_min) / 2
    # gamma = math.sqrt(rho_max * rho_min)

    h0 = h0.intersection(shapely.affinity.scale(h12, xfact=gamma,
                                                yfact=gamma))

    t2 = time.time()
    T_COMP += (t2 - t1)

    NITER += 1
    print(f'{NITER:3.0f}.', f'{rho_min:.6f}', f'{gamma:.6f}', f'{rho_max:.6f}',
          '   ', len(h0.boundary.coords) - 1)
    scale0 = 1 / max(h0.bounds[2], h0.bounds[3])
    h0 = shapely.affinity.scale(h0, xfact=scale0, yfact=scale0)

    if (rho_max - rho_min) < TOL:
        break

# Plotting

h10 = shapely.affinity.scale(h1, xfact=gamma, yfact=gamma)
p10 = np.asarray(MultiPoint(h10.boundary.coords))

h20 = shapely.affinity.scale(h2, xfact=gamma, yfact=gamma)
p20 = np.asarray(MultiPoint(h20.boundary.coords))

bb = 2. * max(h0.bounds[2], h10.bounds[2], h20.bounds[2],
              h0.bounds[3], h10.bounds[3], h20.bounds[3])

pyplot.rc('text', usetex=True)
pyplot.rc('font', family='serif')

# =================================================================
# Tuning the LaTex preamble (e.g. for international support)
#
# pyplot.rcParams['text.latex.preamble'] = \
#     r'\usepackage[utf8]{inputenc}' + '\n' + \
#     r'\usepackage[russian]{babel}' + '\n' + \
#     r'\usepackage{amsmath}'
# =================================================================

fig = pyplot.figure(num="Maximum growth rate trajectory", dpi=108)
ax = fig.add_subplot(111)
ax.set_xlim(-1.1*bb, 1.1*bb)
ax.set_ylim(-1.1*bb, 1.1*bb)
ax.set_aspect(1)
ax.grid(True, linestyle=":")

ax.plot(np.array(p10)[:, 0], np.array(p10)[:, 1], '--',
        color='red', linewidth=1, label=r'$\|A_{1}x\|=\gamma$')
ax.legend()

ax.plot(np.array(p20)[:, 0], np.array(p20)[:, 1], ':',
        color='blue', linewidth=1, label=r'$\|A_{2}x\|=\gamma$')
ax.legend()

ax.plot(np.array(p0)[:, 0], np.array(p0)[:, 1], '-',
        color='black', label=r'$\|x\|=1$')
ax.legend()

# Plotting lines of intersection of norms unit spheres

pl10 = LineString(p10)
pl20 = LineString(p20)
h_int = np.asarray(shapely.affinity.scale(pl10.intersection(pl20),
                                          xfact=3, yfact=3))
for i in range(np.size(h_int[:, 0])):
    if h_int[i, 0] >= 0:
        ax.plot([h_int[i, 0], -h_int[i, 0]], [h_int[i, 1], -h_int[i, 1]], '-',
                color='green', linewidth=0.25)

# Iterations

x = np.asarray([1, 1])
x1 = np.asarray([0, 0])
x2 = np.asarray([0, 0])
A0T = np.transpose(A0)
A1T = np.transpose(A1)

if gamma > 1:
    x = (0.3 / polygonal_norm(x[0], x[1], h0)) * x
else:
    x = (2.0 / polygonal_norm(x[0], x[1], h0)) * x

for i in range(1000):
    xprev = x
    x1 = np.matmul(x, A0T)
    x2 = np.matmul(x, A1T)
    if polygonal_norm(x1[0], x1[1], h0) > polygonal_norm(x2[0], x2[1], h0):
        x = x1
        ax.arrow(xprev[0], xprev[1], x[0]-xprev[0], x[1]-xprev[1],
                 head_width=0.03, head_length=0.06, linewidth=0.5,
                 color='red', length_includes_head=True)
    else:
        x = x2
        ax.arrow(xprev[0], xprev[1], x[0]-xprev[0], x[1]-xprev[1],
                 head_width=0.03, head_length=0.06, linewidth=0.5,
                 color='blue', length_includes_head=True)
    if ((polygonal_norm(x[0], x[1], h0) > 2.) or
            (polygonal_norm(x[0], x[1], h0) < 0.3)):
        break

t3 = time.time()

pyplot.show()
print('\nInitialization: ', f'{round(t00-t0, 6):6.2f} sec.')
print('Computations:   ', f'{round(T_COMP, 6):6.2f} sec.')
print('Plotting:       ', f'{round(t3-t2, 6):6.2f} sec.')
print('Total:          ', f'{round(t3-t0, 6):6.2f} sec.')

print('\nModules used:  Python ' + platform.python_version(),
      'matplotlib ' + version('matplotlib'),
      'numpy ' + version('numpy'),
      'shapely ' + version('shapely'), sep=', ')
