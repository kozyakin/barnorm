# -*- coding: utf-8 -*-
"""Barabanov norms for rotation matrices.

Created on Sat Sep 21 12:37:46 2019.
Last updated on Sun Jul 21 02:22:09 2024 +0300
Make compatible with Shapely v2.0

@author: Victor Kozyakin
"""
import math
import platform
import time
from importlib.metadata import version

import numpy as np
import shapely
import shapely.affinity
from matplotlib import pyplot
from matplotlib.ticker import MultipleLocator
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
    _ll = LineString([(0, 0), (_scale * _x, _scale * _y)])
    _p_int = _ll.intersection(_h).coords
    return math.sqrt((_x**2 + _y**2) / (_p_int[1][0]**2 + _p_int[1][1]**2))


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
    _sg = [1 / polygonal_norm(_pg[i][0], _pg[i][1], _h)
           for i in range(_dimg)]
    _ph = _h.boundary.coords
    _dimh = len(_ph) - 1
    _sh = [polygonal_norm(_ph[i][0], _ph[i][1], _g) for i in range(_dimh)]
    _sgh = _sg + _sh
    return (min(_sgh), max(_sgh))


def matrix_angular_coord(_a, _t):
    """Calculate the angular coordinate of vector Ax given vector x.

    Args:
        _a (2x2 np.array): input matrix A
        _t (nx1 np.array): array of input angles of x's

    Returns:
        [nx1 np.array]: array of output angles of Ax's
    """
    _cos_t = math.cos(_t)
    _sin_t = math.sin(_t)
    _vec_t = np.asarray([_cos_t, _sin_t])
    _vec_t_transpose = np.transpose(_vec_t)
    _rot_back = np.asarray([[_cos_t, _sin_t], [-_sin_t, _cos_t]])
    _vec_a = np.matmul(np.matmul(_rot_back, _a), _vec_t_transpose)
    return _t + math.atan2(_vec_a[1], _vec_a[0])


# Initialization

t_tick = time.time()
t_barnorm_comp = float(0)

TOL = 0.0000001
ANGLE_STEP = 0.01
LEN_TRAJECTORY = 10000
NUM_SYMB = 50
L_BOUND = 0.2
U_BOUND = 2.2

THETA0 = 0.7  # 0.4  # 0.6151 # one point of discontinuity
THETA1 = 0.8
COS_A0 = math.cos(THETA0)
SIN_A0 = math.sin(THETA0)
COS_A1 = math.cos(THETA1)
SIN_A1 = math.sin(THETA1)
LAMBDA = 0.75

A0 = np.asarray([[COS_A0, -SIN_A0], [SIN_A0, COS_A0]])
A1 = np.asarray([[COS_A1, -LAMBDA * SIN_A1],
                [(1 / LAMBDA) * SIN_A1, COS_A1]])
A0T = np.transpose(A0)
A1T = np.transpose(A1)

# Computation initialization

if ((np.linalg.det(A0) == 0) or (np.linalg.det(A1) == 0)):
    raise SystemExit("Set of matrices is degenerate. End of work!")

INV_A0 = np.linalg.inv(A0)
INV_A1 = np.linalg.inv(A1)
INV_A0T = np.transpose(INV_A0)
INV_A1T = np.transpose(INV_A1)

p0 = np.asarray([[1, -1], [1, 1]])
p0 = np.concatenate((p0, -p0), axis=0)
p0 = MultiPoint(p0)
h0 = p0.convex_hull

scale0 = 1 / max(h0.bounds[2], h0.bounds[3])
h0 = shapely.affinity.scale(h0, xfact=scale0, yfact=scale0)

t_ini = time.time() - t_tick

print('\n  #   rho_min    rho    rho_max  Num_edges\n')

# Computation iterations

NITER = 0.
while True:
    t_tick = time.time()

    p0 = np.array(h0.boundary.coords)

    p1 = MultiPoint(np.matmul(p0, INV_A0T))
    h1 = p1.convex_hull

    p2 = MultiPoint(np.matmul(p0, INV_A1T))
    h2 = p2.convex_hull

    h12 = h1.intersection(h2)
    p12 = MultiPoint(h12.boundary.coords)

    rho_minmax = min_max_norms_quotent(h12, h0)
    rho_max = rho_minmax[1]
    rho_min = rho_minmax[0]

    rho = (rho_max + rho_min) / 2

    h0 = h0.intersection(shapely.affinity.scale(h12, xfact=rho, yfact=rho))

    t_barnorm_comp += (time.time() - t_tick)

    NITER += 1
    print(f'{NITER:3.0f}.', f'{rho_min:.6f}', f'{rho:.6f}', f'{rho_max:.6f}',
          '   ', len(h0.boundary.coords) - 1)
    scale0 = 1 / max(h0.bounds[2], h0.bounds[3])
    h0 = shapely.affinity.scale(h0, xfact=scale0, yfact=scale0)

    if (rho_max - rho_min) < TOL:
        break

# Plotting Barabanov norm

t_tick = time.time()

h10 = shapely.affinity.scale(h1, xfact=rho, yfact=rho)
p10 = np.array(h10.boundary.coords)

h20 = shapely.affinity.scale(h2, xfact=rho, yfact=rho)
p20 = np.array(h20.boundary.coords)

bb = 1.7 * max(h0.bounds[2], h10.bounds[2], h20.bounds[2],
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

# Plotting Barabanov's norm

fig1 = pyplot.figure(num="Barabanov norm", dpi=108)
ax1 = fig1.add_subplot(111)
ax1.set_xlim(-1.1 * bb, 1.1 * bb)
ax1.set_ylim(-1.1 * bb, 1.1 * bb)
ax1.set_aspect(1)
ax1.tick_params(labelsize=16)
ax1.grid(True, linestyle=":")
ax1.xaxis.set_major_locator(MultipleLocator(1))
ax1.yaxis.set_major_locator(MultipleLocator(1))

ax1.plot(p10[:, 0], p10[:, 1], ':', color='red', linewidth=1.25)
ax1.plot(p20[:, 0], p20[:, 1], '--', color='blue', linewidth=1)
ax1.plot(p0[:, 0], p0[:, 1], '-', color='black')

# Plotting lines of intersection of norms' unit spheres

pl10 = LineString(p10)
pl20 = LineString(p20)
h_int = shapely.affinity.scale(pl10.intersection(pl20), xfact=3, yfact=3)
p_int = np.array([[pt.x, pt.y] for pt in h_int.geoms])

arr_switch_N = np.size(p_int[:, 0])
arr_switch_ang = np.empty(arr_switch_N)
for i in range(np.size(p_int[:, 0])):
    arr_switch_ang[i] = math.atan2(p_int[i, 1], p_int[i, 0])
    if arr_switch_ang[i] < 0:
        arr_switch_ang[i] = arr_switch_ang[i] + 2. * math.pi
    if p_int[i, 0] >= 0:
        ax1.plot([2 * p_int[i, 0], -2 * p_int[i, 0]],
                 [2 * p_int[i, 1], -2 * p_int[i, 1]],
                 dashes=[5, 2, 1, 2], color='green', linewidth=1)

t_plot_fig1 = time.time() - t_tick
pyplot.show()


# Plotting an extremal trajectory

t_tick = time.time()

fig2 = pyplot.figure(num="Maximum growth rate trajectory", dpi=108)
ax2 = fig2.add_subplot(111)
ax2.set_xlim(-1.1 * bb, 1.1 * bb)
ax2.set_ylim(-1.1 * bb, 1.1 * bb)
ax2.set_aspect(1)
ax2.tick_params(labelsize=16)
ax2.grid(True, linestyle=":")
ax2.xaxis.set_major_locator(MultipleLocator(1))
ax2.yaxis.set_major_locator(MultipleLocator(1))

# Plotting lines of intersection of norms' unit spheres

arr_switch_N = np.size(p_int[:, 0])
arr_switch_ang = np.empty(arr_switch_N)
for i in range(np.size(p_int[:, 0])):
    arr_switch_ang[i] = math.atan2(p_int[i, 1], p_int[i, 0])
    if arr_switch_ang[i] < 0:
        arr_switch_ang[i] = arr_switch_ang[i] + 2. * math.pi
    if p_int[i, 0] >= 0:
        ax2.plot([2 * p_int[i, 0], -2 * p_int[i, 0]],
                 [2 * p_int[i, 1], -2 * p_int[i, 1]],
                 dashes=[5, 2, 1, 2], color='green', linewidth=1)


# Plotting the trajectory

x = np.asarray([1, 1])

if rho > 1:
    x = (L_BOUND / polygonal_norm(x[0], x[1], h0)) * x
else:
    x = (U_BOUND / polygonal_norm(x[0], x[1], h0)) * x

for i in range(LEN_TRAJECTORY):
    xprev = x
    x0 = np.matmul(x, A0T)
    x1 = np.matmul(x, A1T)
    if (polygonal_norm(x0[0], x0[1], h0) >
            polygonal_norm(x1[0], x1[1], h0)):
        x = x0
        ax2.arrow(xprev[0], xprev[1], x[0] - xprev[0], x[1] - xprev[1],
                  head_width=0.04, head_length=0.08, linewidth=0.75,
                  color='red', length_includes_head=True, zorder=-i)
    else:
        x = x1
        ax2.arrow(xprev[0], xprev[1], x[0] - xprev[0], x[1] - xprev[1],
                  head_width=0.04, head_length=0.08, linewidth=0.75,
                  color='blue', length_includes_head=True, zorder=-i)
    if ((polygonal_norm(x[0], x[1], h0) > U_BOUND) or
            (polygonal_norm(x[0], x[1], h0) < L_BOUND)):
        break

arr_switch_ang.sort()
ISPLIT = 0
for i in range(np.size(arr_switch_ang)):
    if arr_switch_ang[i] < math.pi:
        ISPLIT = i

arr_switch_ang = np.resize(arr_switch_ang, ISPLIT + 1)
arr_switch_N = np.size(arr_switch_ang)
arr_switches = np.insert(arr_switch_ang, 0, 0)
arr_switches = np.append(arr_switches, math.pi)
omega1 = (arr_switches[1] + arr_switches[2]) / 2.
omega2 = omega1 + math.pi / 2.
omega3 = omega2 + math.pi / 2.
omega4 = omega3 + math.pi / 2.
props = {'boxstyle': 'round', 'facecolor': 'gainsboro',
         'edgecolor': 'none', 'alpha': 0.5}
p_label = np.array([math.cos(omega1), math.sin(omega1)])

if (polygonal_norm(p_label[0], p_label[1], h10) >
        polygonal_norm(p_label[0], p_label[1], h20)):
    ax2.text(0.9 * bb * math.cos(omega1), 0.9 * bb * math.sin(omega1),
             r'$x_{n+1}=A_0x_n$', ha='center', va='center',
             fontsize='x-large', bbox=props)
    ax2.text(0.8 * bb * math.cos(omega2), 0.8 * bb * math.sin(omega2),
             r'$x_{n+1}=A_1x_n$', ha='center', va='center',
             fontsize='x-large', bbox=props)
    ax2.text(0.9 * bb * math.cos(omega3), 0.9 * bb * math.sin(omega3),
             r'$x_{n+1}=A_0x_n$', ha='center', va='center',
             fontsize='x-large', bbox=props)
    ax2.text(0.8 * bb * math.cos(omega4), 0.8 * bb * math.sin(omega4),
             r'$x_{n+1}=A_1x_n$', ha='center', va='center',
             fontsize='x-large', bbox=props)
else:
    ax2.text(0.8 * bb * math.cos(omega1), 0.8 * bb * math.sin(omega1),
             r'$x_{n+1}=A_1x_n$', ha='center', va='center',
             fontsize='x-large', bbox=props)
    ax2.text(0.9 * bb * math.cos(omega2), 0.9 * bb * math.sin(omega2),
             r'$x_{n+1}=A_0x_n$', ha='center', va='center',
             fontsize='x-large', bbox=props)
    ax2.text(0.8 * bb * math.cos(omega3), 0.8 * bb * math.sin(omega3),
             r'$x_{n+1}=A_1x_n$', ha='center', va='center',
             fontsize='x-large', bbox=props)
    ax2.text(0.9 * bb * math.cos(omega4), 0.9 * bb * math.sin(omega4),
             r'$x_{n+1}=A_0x_n$', ha='center', va='center',
             fontsize='x-large', bbox=props)

t_plot_fig2 = time.time() - t_tick
pyplot.show()

# Plotting the angular functions

t_tick = time.time()

fig3 = pyplot.figure(num="Angular function", dpi=108)
ax3 = fig3.add_subplot(111)
ax3.set_xlim(0., math.pi)
ax3.set_ylim(0., math.pi)
ax3.set_aspect(1)
ax3.tick_params(labelsize=16)

t = np.arange(0., math.pi, ANGLE_STEP)
angle_arr_A0 = np.empty(len(t))
angle_arr_A1 = np.empty(len(t))
for i, item in enumerate(t):
    angle_arr_A0[i] = matrix_angular_coord(A0, item)
    angle_arr_A1[i] = matrix_angular_coord(A1, item)

ax3.plot(t, angle_arr_A0, linestyle=(0, (30, 30)), color='red',
         linewidth=0.15)
ax3.plot(t, angle_arr_A0 + math.pi, linestyle=(0, (30, 30)), color='red',
         linewidth=0.15)
ax3.plot(t, angle_arr_A0 - math.pi, linestyle=(0, (30, 30)), color='red',
         linewidth=0.15)
ax3.plot(t, angle_arr_A1, linestyle=(0, (30, 30)), color='blue',
         linewidth=0.15)
ax3.plot(t, angle_arr_A1 + math.pi, linestyle=(0, (30, 30)), color='blue',
         linewidth=0.15)
ax3.plot(t, angle_arr_A1 - math.pi, linestyle=(0, (30, 30)), color='blue',
         linewidth=0.15)

# Plotting the angular function delivering
# the maximal growth rate of iterations

for j in range(arr_switch_N + 1):
    t = np.arange(arr_switches[j], arr_switches[j + 1], ANGLE_STEP)
    angle_arr_A0 = np.empty(len(t))
    angle_arr_A1 = np.empty(len(t))
    for i, item in enumerate(t):
        angle_arr_A0[i] = matrix_angular_coord(A0, item)
        angle_arr_A1[i] = matrix_angular_coord(A1, item)
    omega = (arr_switches[j] + arr_switches[j + 1]) / 2.
    x = np.asarray([math.cos(omega), math.sin(omega)])
    x0 = np.matmul(x, A0T)
    x1 = np.matmul(x, A1T)
    if (polygonal_norm(x0[0], x0[1], h0) <
            polygonal_norm(x1[0], x1[1], h0)):
        ax3.plot(t, angle_arr_A1, 'b', linewidth=1.5)
        ax3.plot(t, angle_arr_A1 + math.pi, 'b', linewidth=1.5)
        ax3.plot(t, angle_arr_A1 - math.pi, 'b', linewidth=1.5)
    else:
        ax3.plot(t, angle_arr_A0, 'r', linewidth=1.5)
        ax3.plot(t, angle_arr_A0 + math.pi, 'r', linewidth=1.5)
        ax3.plot(t, angle_arr_A0 - math.pi, 'r', linewidth=1.5)

# Putting Pi-ticks on axes

xtick_pos = [0, arr_switches[1], 0.5 * np.pi, arr_switches[2], np.pi]
xlabels = [r'0', r'$\omega_0$', '', r'$\omega_1$', r'$\pi$']
ytick_pos = [0, 0.5 * np.pi, np.pi]
ylabels = [r'0', r'$\frac{\pi}{2}$', r'$\pi$']

pyplot.xticks(xtick_pos, xlabels)
pyplot.yticks(ytick_pos, ylabels)
pyplot.grid(linestyle=":")

t_plot_fig3 = time.time() - t_tick
pyplot.show()

# Calculating index sequence

t_tick = time.time()

F0 = 0.
F1 = 0.
F00 = 0.
F01 = 0.
F10 = 0.
F11 = 0.
x = np.asarray([1, 1])
index_seq = []

for i in range(LEN_TRAJECTORY):
    x = x / polygonal_norm(x[0], x[1], h0)
    x0 = np.matmul(x, A0T)
    x1 = np.matmul(x, A1T)
    if (polygonal_norm(x0[0], x0[1], h0) >
            polygonal_norm(x1[0], x1[1], h0)):
        x = x0
        index_seq.append('0')
        F0 += 1
    else:
        x = x1
        index_seq.append('1')
        F1 += 1
    if i > 0:
        if ((index_seq[i - 1] == '0') and (index_seq[i] == '0')):
            F00 += 1
        if ((index_seq[i - 1] == '0') and (index_seq[i] == '1')):
            F01 += 1
        if ((index_seq[i - 1] == '1') and (index_seq[i] == '0')):
            F10 += 1
        if ((index_seq[i - 1] == '1') and (index_seq[i] == '1')):
            F11 += 1

print('\nExtremal index sequence: ', end='')
for i in range(NUM_SYMB):
    print(index_seq[i], end='')

print('\n\nFrequences of symbols 0, 1, 00, 01 etc. in the index sequence:',
      '\n\nSymbols:       0      1      00     01     10     11')

print('Frequences: ',
      f' {round(F0 / LEN_TRAJECTORY, 3):.3f}',
      f' {round(F1 / LEN_TRAJECTORY, 3):.3f}',
      f' {round(F00 / (LEN_TRAJECTORY - 1), 3):.3f}',
      f' {round(F01 / (LEN_TRAJECTORY - 1), 3):.3f}',
      f' {round(F10 / (LEN_TRAJECTORY - 1), 3):.3f}',
      f' {round(F11 / (LEN_TRAJECTORY - 1), 3):.3f}')

t_index_seq = time.time() - t_tick

# Saving plots to pdf-files

"""
fig1.savefig(f'bnorm-{THETA0:.2f}-{THETA1:.2f}-{LAMBDA:.2f}.pdf',
             bbox_inches='tight')
fig2.savefig(f'etraj-{THETA0:.2f}-{THETA1:.2f}-{LAMBDA:.2f}.pdf',
             bbox_inches='tight')
fig3.savefig(f'sfunc-{THETA0:.2f}-{THETA1:.2f}-{LAMBDA:.2f}.pdf',
             bbox_inches='tight')
"""

# Computation timing

t_compute = t_barnorm_comp + t_index_seq
t_plot = t_plot_fig1 + t_plot_fig2 + t_plot_fig3
t_total = t_ini + t_plot + t_compute


print('\nInitialization: ', f'{round(t_ini, 6):6.2f} sec.')
print('Computations:   ', f'{round(t_compute, 6):6.2f} sec.')
print('Plotting:       ', f'{round(t_plot, 6):6.2f} sec.')
print('Total:          ', f'{round(t_total, 6):6.2f} sec.')

print('\nModules used:  Python ' + platform.python_version(),
      'matplotlib ' + version('matplotlib'),
      'numpy ' + version('numpy'),
      'shapely ' + version('shapely'), sep=', ')
