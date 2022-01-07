# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 12:37:46 2019.
Last updated on Mon Jan 3 14:39:32 2022.

@author: Victor Kozyakin
"""
import time
import math
from matplotlib import pyplot
import numpy as np
import shapely
from shapely.geometry import LineString
from shapely.geometry import MultiPoint


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
    _scale = 0.5 * math.sqrt(
        ((_hb[2] - _hb[0])**2 +
         (_hb[3] - _hb[1])**2) / (_x**2 + _y**2))
    _ll = LineString([(0, 0), (_scale*_x, _scale*_y)])
    _h_int = _ll.intersection(_h).coords
    return math.sqrt(
        (_x**2 + _y**2) / (_h_int[1][0]**2 + _h_int[1][1]**2))


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
    _sh = [polygonal_norm(_ph[i][0], _ph[i][1], _g)
           for i in range(_dimh)]
    _sgh = _sg + _sh
    return (min(_sgh), max(_sgh))

# Initialization


t_tick = time.time()
t_Barnorm_comp = 0.

TOL = 0.0000001
ANGLE_STEP = 0.01
LEN_TRAJECTORY = 10000
NUM_SYMB = 50
L_BOUND = 0.3
U_BOUND = 2.

THETA0 = 0.9
THETA1 = 0.8
COS_A0 = math.cos(THETA0)
COS_A1 = math.cos(THETA1)

A0 = np.asarray([[2. * COS_A0, -1.], [1., 0.]])
A1 = np.asarray([[0., 1.], [-1., 2. * COS_A1]])
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

niter = 0.
while True:
    t_tick = time.time()

    p0 = np.asarray(MultiPoint(h0.boundary.coords))

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
    # rho = math.sqrt(rho_max * rho_min)

    h0 = h0.intersection(shapely.affinity.scale(h12, xfact=rho,
                                                yfact=rho))

    t_Barnorm_comp += (time.time() - t_tick)

    niter += 1
    print(f'{niter:3.0f}.', f'{rho_min:.6f}',
          f'{rho:.6f}', f'{rho_max:.6f}', '   ',
          len(h0.boundary.coords) - 1)
    scale0 = 1 / max(h0.bounds[2], h0.bounds[3])
    h0 = shapely.affinity.scale(h0, xfact=scale0, yfact=scale0)

    if (rho_max - rho_min) < TOL:
        break

# Plotting Barabanov norm

t_tick = time.time()

h10 = shapely.affinity.scale(h1, xfact=rho, yfact=rho)
p10 = np.asarray(MultiPoint(h10.boundary.coords))

h20 = shapely.affinity.scale(h2, xfact=rho, yfact=rho)
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

fig = pyplot.figure(1, dpi=108)
ax = fig.add_subplot(111)
ax.set_xlim(-1.1*bb, 1.1*bb)
ax.set_ylim(-1.1*bb, 1.1*bb)
ax.set_aspect(1)
ax.tick_params(labelsize=16)
ax.grid(True, linestyle=":")

ax.plot(p10[:, 0], p10[:, 1], '--',
        color='red', linewidth=1, label=r'$\|A_{0}x\|=\rho$')
ax.legend()

ax.plot(p20[:, 0], p20[:, 1], '--',
        color='blue', linewidth=1, label=r'$\|A_{1}x\|=\rho$')
ax.legend()

ax.plot(p0[:, 0], p0[:, 1], '-',
        color='black', label=r'$\|x\|=1$')
ax.legend()

# Plotting lines of intersection of norms' unit spheres

pl10 = LineString(p10)
pl20 = LineString(p20)
h_int = np.asarray(shapely.affinity.scale(pl10.intersection(pl20),
                                          xfact=3, yfact=3))
arr_switch_N = np.size(h_int[:, 0])
arr_switch_ang = np.empty(arr_switch_N)
for i in range(np.size(h_int[:, 0])):
    arr_switch_ang[i] = math.atan2(h_int[i, 1], h_int[i, 0])
    if arr_switch_ang[i] < 0:
        arr_switch_ang[i] = arr_switch_ang[i] + 2. * math.pi
    if h_int[i, 0] >= 0:
        ax.plot([h_int[i, 0], -h_int[i, 0]],
                [h_int[i, 1], -h_int[i, 1]],
                '-', color='green', linewidth=0.25)

ax.plot(np.NaN, np.NaN, '-', color='green', linewidth=0.25,
        label=r'$\|A_{0}x\|=\|A_{1}x\|$')
ax.legend()

arr_switch_ang.sort()
isplit = 0
for i in range(np.size(arr_switch_ang)):
    if arr_switch_ang[i] < math.pi:
        isplit = i

arr_switch_ang = np.resize(arr_switch_ang, isplit + 1)
arr_switch_N = np.size(arr_switch_ang)

# Plotting extremal trajectory

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
        ax.arrow(xprev[0], xprev[1], x[0]-xprev[0], x[1]-xprev[1],
                 head_width=0.03, head_length=0.07, linewidth=0.75,
                 color='red', length_includes_head=True)
    else:
        x = x1
        ax.arrow(xprev[0], xprev[1], x[0]-xprev[0], x[1]-xprev[1],
                 head_width=0.03, head_length=0.07, linewidth=0.75,
                 color='blue', length_includes_head=True)
    if ((polygonal_norm(x[0], x[1], h0) > U_BOUND) or
            (polygonal_norm(x[0], x[1], h0) < L_BOUND)):
        break

t_traj_plot = time.time() - t_tick
pyplot.show()


# Plotting the angle functions

t_tick = time.time()


def matrix_angular_coord(_a, _t):
    """Calculation of angular coordinate for vector Ax, given
    angular coordinate of vector x

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
    _rot_back = np.asarray([[_cos_t, _sin_t],  [-_sin_t, _cos_t]])
    _vec_a = np.matmul(np.matmul(_rot_back, _a), _vec_t_transpose)
    return _t + math.atan2(_vec_a[1], _vec_a[0])


fig2 = pyplot.figure(2, dpi=108)
ax1 = fig2.add_subplot(111)
ax1.set_xlim(0., math.pi)
ax1.set_ylim(0., math.pi)
ax1.set_aspect(1)
ax1.tick_params(labelsize=16)

for i in range(arr_switch_N):
    ax1.plot([arr_switch_ang[i], arr_switch_ang[i]], [0, math.pi],
             '-', color="green", linewidth=0.5)

t = np.arange(0., math.pi, ANGLE_STEP)
angle_arr_A0 = np.empty(len(t))
angle_arr_A1 = np.empty(len(t))
for i, item in enumerate(t):
    angle_arr_A0[i] = matrix_angular_coord(A0, item)
    angle_arr_A1[i] = matrix_angular_coord(A1, item)

ax1.plot(t, t, 'g--',
         t, angle_arr_A0, 'r--',
         t, angle_arr_A1, 'b--', linewidth=0.15)
ax1.plot(t, angle_arr_A0 + math.pi, 'r--',
         t, angle_arr_A1 + math.pi, 'b--', linewidth=0.15)
ax1.plot(t, angle_arr_A0 - math.pi, 'r--',
         t, angle_arr_A1 - math.pi, 'b--', linewidth=0.15)

# Plotting the angle function delivering
# the maximal growth rate of iterations

arr_switches = np.insert(arr_switch_ang, 0, 0)
arr_switches = np.append(arr_switches, math.pi)

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
        ax1.plot(t, angle_arr_A1, 'b', linewidth=1.5)
        ax1.plot(t, angle_arr_A1 + math.pi, 'b', linewidth=1.5)
        ax1.plot(t, angle_arr_A1 - math.pi, 'b', linewidth=1.5)
    else:
        ax1.plot(t, angle_arr_A0, 'r', linewidth=1.5)
        ax1.plot(t, angle_arr_A0 + math.pi, 'r', linewidth=1.5)
        ax1.plot(t, angle_arr_A0 - math.pi, 'r', linewidth=1.5)

# Put Pi-ticks on axes

xtick_pos = [0, arr_switches[1], 0.5 * np.pi, arr_switches[2],
             np.pi]
xlabels = [r'0', r'$\omega_0$', r'$\frac{\pi}{2}$',
           r'$\omega_1$', r'$\pi$']

ytick_pos = [0, 0.5 * np.pi, np.pi]
ylabels = [r'0', r'$\frac{\pi}{2}$', r'$\pi$']

pyplot.xticks(xtick_pos, xlabels)
pyplot.yticks(ytick_pos, ylabels)
pyplot.grid(linestyle=":")

t_plot_ang_fun = time.time() - t_tick
pyplot.show()


# Calculating index sequence

t_tick = time.time()

f0 = 0.
f1 = 0.
x = np.asarray([1, 1])

print('\nExtremal index sequence: ', end='')
for i in range(LEN_TRAJECTORY):
    x = x / polygonal_norm(x[0], x[1], h0)
    x0 = np.matmul(x, A0T)
    x1 = np.matmul(x, A1T)
    if (polygonal_norm(x0[0], x0[1], h0) >
            polygonal_norm(x1[0], x1[1], h0)):
        x = x0
        f0 += 1
        if i < NUM_SYMB:
            print('0', end='')
    else:
        x = x1
        f1 += 1
        if i < NUM_SYMB:
            print('1', end='')

print(f'\n\nFreq_of_0 = {round(f0/LEN_TRAJECTORY, 3):.3f},',
      f' freq_of_1 = {round(f1/LEN_TRAJECTORY, 3):.3f}')
t_index_seq = time.time() - t_tick


# Saving plots to pdf-files

"""
fig.savefig(f'barnorm-{THETA0:.2f}-{THETA1:.2f}.pdf',
            bbox_inches='tight')
fig2.savefig(f'anglefun-{THETA0:.2f}-{THETA1:.2f}.pdf',
             bbox_inches='tight')
"""

# Computation timing

t_total = (t_ini + t_plot_ang_fun + t_traj_plot + t_Barnorm_comp +
           t_index_seq)

print('\n')
print('Initialization: ' +
      f'{round(t_ini, 6):6.2f} sec.')
print('Computations:   ' +
      f'{round(t_Barnorm_comp + t_index_seq, 6):6.2f} sec.')
print('Plotting:       ' +
      f'{round(t_traj_plot + t_plot_ang_fun, 6):6.2f} sec.')
print('Total:          ' +
      f'{round(t_total, 6):6.2f} sec.')
