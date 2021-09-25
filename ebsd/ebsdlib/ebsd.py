# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 17:14:46 2021

@author: harini
"""
import math
import linecache
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from skimage.filters import sobel, threshold_otsu, threshold_multiotsu, threshold_triangle
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import color
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops
from numba import njit
import time
import numba as nb
import warnings

# for cube
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.spatial.transform import Rotation as R

# import cube_with_pole_figure
# for pole figure and ipf
# import pf
###########
__all__ = ['read_data', 'medianFilter', 'rgb_img', 'find_edges', 'watersheding',
           'get_cube', 'get_correct_quivers', 'rotateZXZ', 'get_quivers', 'get_m',
           'rotation', 'stereographicProjection_001', 'stereographicProjection_100',
           'stereographicProjection_010', 'draw_circle_frame', 'draw_wulff_net',
           'draw_trace', 'pf_map', 'rotateZXZ_ipf', 'theta', 'kam', 'dislocation_density_map',
           'get_IPF_color', 'ipfmap', 'kam_segmentation', 'grod', 'ipf_legend']


def read_data(file):
    '''
    Parameters
    ----------
    file : STR
        path of data file in the system.

    Returns
    -------
    Eulers : numpy.ndarray (3,rows,cols)
        Euler angle array.
    data0 : numpy.ndarray
    cols : int
    rows : int
    xStep : float
    yStep : float
    '''
    f = open(file)
    prj = str(linecache.getline(file, 2))
    cols = int(linecache.getline(file, 5)[7:])
    rows = int(linecache.getline(file, 6)[7:])
    xStep = float(linecache.getline(file, 7)[6:])
    yStep = float(linecache.getline(file, 8)[6:])
    Phases = int(linecache.getline(file, 13)[7:])
    skipRows = 14+Phases
    data1 = np.transpose(np.loadtxt(file, skiprows=skipRows))
    data0 = data1[0].reshape(rows, cols)
    Eulers = np.zeros((3, rows, cols))
    Eulers[0][:][:] = data1[5].reshape(rows, cols)
    Eulers[1][:][:] = data1[6].reshape(rows, cols)
    Eulers[2][:][:] = data1[7].reshape(rows, cols)
    f.close()

    return(Eulers, data0, cols, rows, xStep, yStep, prj)


@njit
def medianFilter(Eulers, data0, w):
    '''
    Returns Median Filtered Euler array
    Parameters
    ----------
    Eulers : numpy.ndarray (3,rows,cols)
    data0 : numpy.ndarray
    w : INT, optional
        kernel size. The default is 3.

    Returns
    -------
    Eulers : numpy.ndarray
    '''
    for i in range(w, Eulers.shape[1]-w):
        for j in range(w, Eulers.shape[2]-w):
            if data0[i, j] == 0:
                block0 = Eulers[0, i-w:i+w+1, j-w:j+w+1]
                m0 = np.median(block0)
                block1 = Eulers[1, i-w:i+w+1, j-w:j+w+1]
                m1 = np.median(block1)
                block2 = Eulers[2, i-w:i+w+1, j-w:j+w+1]
                m2 = np.median(block2)
                Eulers[0, i, j] = m0
                Eulers[1, i, j] = m1
                Eulers[2, i, j] = m2
    return Eulers


def rgb_img(Eulers):
    '''
    Returns an Image of all Eulers
    Returns
    -------
    rgb : np.uint8

    '''
    e1, e2, e3 = Eulers[0, :, :], Eulers[1, :, :], Eulers[2, :, :]
    e11 = (255/360)*e1
    e21 = (255/90)*e2
    e31 = (255/90)*e3
    rgb = np.zeros((e1.shape[0], e1.shape[1], 3))
    rgb[:, :, 0] = e11
    rgb[:, :, 1] = e21
    rgb[:, :, 2] = e31
    rgb = np.uint8(rgb)
    return rgb


def find_edges(filtered_img):
    '''
    Parameters
    ----------
    filtered_img : 2D array (0, rows, cols)

    Returns
    -------
    multi : image with edges
    binary : binary image with edges.

    '''
    sobel_img = sobel(filtered_img)
    t2 = threshold_triangle(sobel_img)
    binary = sobel_img < t2
    multi = binary * filtered_img
    return multi, binary


'''Step 1: Read image img_cells.
Step 2: Make a binary image were the cells are forground and the rest is background(edges).
Step 3: Fill interior gaps if necessary.
Step 4: Compute the distance function of the cell regions to the background.
Step 5: Compute the local maxima of the distance transform.
Step 6: Use the local maxima as markers, and the negative distance transform as segmentation function, and compute the watershed transform.'''


def watersheding(multi):
    '''
    Parameters
    ----------
    multi : image with edges

    Returns
    -------
    wc_img : watersheded image with colored labels.
    labels : only labels
    n : number of regions identified.

    '''
    # Compute the (euclidean) distance transform of the complement of the binary image.
    # Generate the markers as local maxima of the distance to the background
    distance = nd.distance_transform_edt(multi)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)))
    # Create markers to be used as seeds for the watershed at the local maxima in the distance transformed image.
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, n = nd.label(distance)
    labels = watershed(-distance, markers, mask=multi, compactness=0.0001)
    wc_img = color.label2rgb(labels, bg_label=0)
    return wc_img, labels, n


# cube functionss

def get_cube():
    phi = np.arange(1, 10, 2)*np.pi/4
    Phi, Theta = np.meshgrid(phi, phi)

    x = np.cos(Phi)*np.sin(Theta)
    y = np.sin(Phi)*np.sin(Theta)
    z = np.cos(Theta)/np.sqrt(2)
    return x, y, z


def get_correct_quivers(qx, qy, qz, pole='001'):
    '''
    Parameters
    ----------
   qx : np.array
        quiver x.
    qy : np.array
        quiver y.
    qz : np.array
        quiver z.
    pole : str, optional
        takes the axis of pole figure. Default value '001'.

    Returns
    -------
    qx : np.array
        quiver x.
    qy : np.array
        quiver y.
    qz : np.array
        quiver z.
    '''
    if(pole == '100'):
        if(qx[0] < 0):
            qx *= -1
        if(qy[0] < 0):
            qy *= -1
        if(qz[0] < 0):
            qz *= -1
    if(pole == '010'):
        if(qx[1] < 0):
            qx *= -1
        if(qy[1] < 0):
            qy *= -1
        if(qz[1] < 0):
            qz *= -1
    if(pole == '001'):
        if(qx[2] < 0):
            qx *= -1
        if(qy[2] < 0):
            qy *= -1
        if(qz[2] < 0):
            qz *= -1
    return qx, qy, qz


def rotateZXZ(m, e1, e2, e3):
    rz2 = np.array([[np.cos(e3), np.sin(e3), 0],
                    [-np.sin(e3), np.cos(e3), 0],
                    [0, 0, 1]])
    rx = np.array([[1, 0, 0],
                   [0, np.cos(e2), np.sin(e2)],
                   [0, -np.sin(e2), np.cos(e2)]])
    rz1 = np.array([[np.cos(e1), np.sin(e1), 0],
                    [-np.sin(e1), np.cos(e1), 0],
                    [0, 0, 1]])
    g1 = rz1 @ rx @ rz2
    return g1 @ m


@njit
def get_quivers():
    qx = np.array([1, 0, 0])
    qy = np.array([0, 1, 0])
    qz = np.array([0, 0, 1])
    return qx, qy, qz


@njit()
def get_m(x, y, z):
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)
    m = np.stack((x, y, z))
    return m


def rotation(m, e1, e2, e3, qx, qy, qz):
    e1, e2, e3 = np.radians(e1), np.radians(e2), np.radians(e3)
    rot = rotateZXZ(m, e1, e2, e3)
    qx = rotateZXZ(qx, e1, e2, e3)
    qy = rotateZXZ(qy, e1, e2, e3)
    qz = rotateZXZ(qz, e1, e2, e3)
    return rot, qx, qy, qz

# pole figure functions


def stereographicProjection_001(qx, qy, qz):
    qx, qy, qz = get_correct_quivers(qx, qy, qz, '001')
    x1, y1, z1 = qx[0], qx[1], qx[2]
    x2, y2, z2 = qy[0], qy[1], qy[2]
    x3, y3, z3 = qz[0], qz[1], qz[2]
    try:
        pf1 = (x1/(1+z1), y1/(1+z1))
        pf2 = (x2/(1+z2), y2/(1+z2))
        pf3 = (x3/(1+z3), y3/(1+z3))
    except ZeroDivisionError:
        pass  # for the pole point (0,0,-1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
    return pf1, pf2, pf3


def stereographicProjection_100(qx, qy, qz):
    qx, qy, qz = get_correct_quivers(qx, qy, qz, '100')
    x1, y1, z1 = qx[0], qx[1], qx[2]
    x2, y2, z2 = qy[0], qy[1], qy[2]
    x3, y3, z3 = qz[0], qz[1], qz[2]
    try:
        pf1 = (y1/(1+x1), z1/(1+x1))
        pf2 = (y2/(1+x2), z2/(1+x2))
        pf3 = (y3/(1+x3), z3/(1+x3))
    except ZeroDivisionError:
        pass  # for the pole point (-1,0,0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
    return pf1, pf2, pf3


def stereographicProjection_010(qx, qy, qz):
    qx, qy, qz = get_correct_quivers(qx, qy, qz, '010')
    x1, y1, z1 = qx[0], qx[1], qx[2]
    x2, y2, z2 = qy[0], qy[1], qy[2]
    x3, y3, z3 = qz[0], qz[1], qz[2]
    try:
        pf1 = (x1/(1+y1), z1/(1+y1))
        pf2 = (x2/(1+y2), z2/(1+y2))
        pf3 = (x3/(1+y3), z3/(1+y3))
    except ZeroDivisionError:
        pass  # for the pole point (0,-1,0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
    return pf1, pf2, pf3

# stereonet functions for pf map


def draw_circle_frame(ax, **kwargs):
    kw = dict(c='k')
    kw.update(kwargs)
    t = np.linspace(0, 2*np.pi, 360)
    ax.plot(np.cos(t), np.sin(t), **kw)
    ax.plot([-1, 1], [0, 0], **kw)
    ax.plot([0, 0], [-1, 1], **kw)


def draw_wulff_net(ax, step=9., theta=0., n=None, **kwargs):
    '''
    Parameters
    ----------
    ax : matplotlib.pyplost figure axis
    step : angle between two adjacent traces
    theta : azimuthal angle
    Returns
    -------
    ax : matplotlib.pyplost figure axis
    '''
    if n:
        step = 180./n
    theta = theta*np.pi/180.
    ctheta, stheta = np.cos(theta), np.sin(theta)

    kw = dict(c='k', lw='.5')
    kw.update(kwargs)
    for t in np.arange(-90., 90., step)[1:]*np.pi/180.:
        # lattitude traces
        draw_trace(ax, n=[-stheta, ctheta, 0], offset=np.sin(t), **kw)
        draw_trace(ax, n=[ctheta*np.cos(t), stheta*np.cos(t),
                          np.sin(t)], **kw)  # longitude traces
    ax.set_axis_off()
    return ax


def draw_trace(ax, n=[1, 0, 1], offset=0., r=1., p=1., **kwargs):
    n = np.asarray(n)/np.linalg.norm(n)  # normalize n
    C = np.zeros(3) + n*offset
    C = C.reshape(3, 1)
    r_prime = (r**2. - offset**2.)**.5  # Radius of the circle

    if (n[0] == 0.) & (n[1] == 0.):
        a = np.asarray([1, 0, 0]).reshape(3, 1)
        b = np.asarray([0, 1, 0]).reshape(3, 1)
    else:
        a = np.asarray([-n[1], n[0], 0]).reshape(3, 1)
        a = a/np.linalg.norm(a)
        b = np.asarray([-n[0]*n[2], -n[1]*n[2], n[0]**2+n[1]**2]).reshape(3, 1)
        b = b/np.linalg.norm(b)

    t1, t2 = 0., 2*np.pi
    if b[2]*r_prime != 0.:
        s = float(C[2]/(b[2]*r_prime))
        if np.abs(s) < 1.:
            t1, t2 = -np.arcsin(s), np.arcsin(s) + np.pi
            if np.sin((t1+t2)/2.) < 0.:
                t1 += 2*np.pi

    t = np.linspace(t1, t2, 50)
    P = r_prime*np.cos(t)*a + r_prime*np.sin(t)*b + C
    # stereographic projection of the circle (trace)
    xp, yp = p*P[0]/(p+P[2]), p*P[1]/(p+P[2])
    ax.plot(xp, yp, **kwargs)

    return ax


def pf_map(Eulers, pfch='{100}'):
    pole = {'{100}': stereographicProjection_100,
            '{010}': stereographicProjection_010,
            '{001}': stereographicProjection_001}
    func = pole[pfch]
    e1, e2, e3 = Eulers[0, :, :], Eulers[1, :, :], Eulers[2, :, :]
    x = np.array([1., 0., 0.])
    y = np.array([0., 1., 0.])
    z = np.array([0., 0., 1.])
    pfX, pfY = list(), list()
    for i in np.arange(e1.shape[0]):
        for j in np.arange(e1.shape[1]):
            a, b, c = e1[i, j], e2[i, j], e3[i, j]
            qx, qy, qz = get_quivers()
            m = get_m(x, y, z)
            m, x, y, z = rotation(m, a, b, c, qx, qy, qz)
            pf1, pf2, pf3 = func(x, y, z)
            # ,s = r[50])#,label='n1',color='red')
            pfX.extend([pf1[0], pf2[0], pf3[0]])
            pfY.extend([pf1[1], pf2[1], pf3[1]])
    return(pfX, pfY)


@njit
def rotateZXZ_ipf(e1, e2, e3):
    '''
    Returns the rotation matrix given 3 euler angles.

    Returns
    -------
    g1 : rotation matrix (3x3)

    '''
    x1 = (np.cos(e1)*np.cos(e3) - np.cos(e2)*np.sin(e1)*np.sin(e3))
    x2 = (np.cos(e3)*np.sin(e1) + np.cos(e2)*np.cos(e1)*np.sin(e3))
    x3 = (np.sin(e2)*np.sin(e3))
    x4 = (-1*np.cos(e1)*np.sin(e3) - np.cos(e2)*np.cos(e3)*np.sin(e1))
    x5 = (np.cos(e2)*np.cos(e1)*np.cos(e3) - np.sin(e1)*np.sin(e3))
    x6 = (np.cos(e3)*np.sin(e2))
    x7 = np.sin(e2)*np.sin(e1)
    x8 = (-1*np.cos(e1)*np.sin(e2))
    x9 = np.cos(e2)
    g1 = np.array([[x1, x2, x3], [x4, x5, x6], [x7, x8, x9]])
    return g1


@njit(fastmath=True)
def theta(p1, p, p2, q1, q, q2):
    x1 = (math.cos(p1)*math.cos(p2) - math.cos(p)*math.sin(p1)*math.sin(p2)) * \
        (math.cos(q1)*math.cos(q2) - math.cos(q)*math.sin(q1)*math.sin(q2))
    x2 = (math.cos(p2)*math.sin(p1) + math.cos(p)*math.cos(p1)*math.sin(p2)) * \
        (math.cos(q2)*math.sin(q1) + math.cos(q)*math.cos(q1)*math.sin(q2))
    x3 = (math.sin(p)*math.sin(p2))*(math.sin(q)*math.sin(q2))
    x4 = (-1*math.cos(p1)*math.sin(p2) - math.cos(p)*math.cos(p2)*math.sin(p1)) * \
        (-1*math.cos(q1)*math.sin(q2) - math.cos(q)*math.cos(q2)*math.sin(q1))
    x5 = (math.cos(p)*math.cos(p1)*math.cos(p2) - math.sin(p1)*math.sin(p2)) * \
        (math.cos(q)*math.cos(q1)*math.cos(q2) - math.sin(q1)*math.sin(q2))
    x6 = (math.cos(p2)*math.sin(p))*(math.cos(q2)*math.sin(q))
    x7 = math.sin(p)*math.sin(p1)*(math.sin(q)*math.sin(q1))
    x8 = (-1*math.cos(p1)*math.sin(p))*(-1*math.cos(q1)*math.sin(q))
    x9 = math.cos(p) * math.cos(q)
    f = 0.5*(x1+x2+x3+x4+x5+x6+x7+x8+x9-1)
    if f > 1 or f < -1:
        f = 0
    return (abs((math.acos(f)))*180/np.pi)


@njit(fastmath=True)
def kam(Eulers):
    e1 = Eulers[0, :, :]*math.pi/180
    e2 = Eulers[1, :, :]*math.pi/180
    e3 = Eulers[2, :, :]*math.pi/180
    y, x = e1.shape[0], e1.shape[1]
    kam = np.zeros((y, x), dtype=np.float64)
    for i in range(y-2):
        for j in range(x-2):
            for m in range(3):
                for n in range(3):
                    if(m == 1 and n == 1):
                        pass
                    elif(m+n) % 2 == 0:
                        kam[i+1, j+1] += theta(e1[i+1, j+1], e2[i+1, j+1], e3[i+1, j+1],
                                               e1[i+m, j+n], e2[i+m, j+n], e3[i+m, j+n])/np.sqrt(2)
                    else:
                        kam[i+1, j+1] += theta(e1[i+1, j+1], e2[i+1, j+1],
                                               e3[i+1, j+1], e1[i+m, j+n], e2[i+m, j+n], e3[i+m, j+n])

    return kam/8


def dislocation_density_map(kam, l, b=235):
    kam = np.radians(kam)
    b *= 10 ** -10
    d = kam/(b*l)
    return d*180/np.pi


@njit
def get_IPF_color(uvw, issorted=False):
    shape = np.shape(uvw)
    ndim = uvw.ndim

    if issorted == False:
        uvw = np.abs(uvw)
        uvw = np.sort(uvw)
        # Select uvw where w >= u >= v
        uvw = np.array([uvw[1], uvw[0], uvw[2]])

    R, G, B = uvw[2] - uvw[0], uvw[0] - uvw[1], uvw[1]

    whitespot = np.array([0.48846011, 0.22903335, 0.84199195])
    pwr = .75

    # Select variant where w >= u >= v
    whitespot = np.sort(whitespot)
    whitespot = whitespot[np.array([1, 0, 2])]

    kR = whitespot[2] - whitespot[0]
    kG = whitespot[0] - whitespot[1]
    kB = whitespot[1]
    R = (R/kR)**pwr
    G = (G/kG)**pwr
    B = (B/kB)**pwr
    rgb = np.array([R, G, B])
    rgbmax = np.max(rgb)
    rgb = rgb*255/rgbmax
    rgb = rgb.astype(np.uint8).T
    if ndim != 2:
        rgb = rgb.reshape(shape)
    return rgb


@njit
def ipfmap(Eulers, ipf_ax='Z'):
    e1, e2, e3 = Eulers[0, :, :], Eulers[1, :, :], Eulers[2, :, :]
    if ipf_ax == 'Z':
        d = np.array([0., 0., 1.])
    elif ipf_ax == 'Y':
        d = np.array([0., 1., 0.])
    else:
        d = np.array([1., 0., 0.])

    ipf_img = np.zeros((e1.shape[0], e1.shape[1], 3), dtype=np.uint8)
    for i in range(e1.shape[0]):
        for j in range(e1.shape[1]):
            phi1, phi, phi2 = np.radians(e1[i, j]), np.radians(
                e2[i, j]), np.radians(e3[i, j])
            M = rotateZXZ_ipf(phi1, phi, phi2)
            color = get_IPF_color(np.dot(M, d))
            ipf_img[i, j, :] = color
    return ipf_img


def kam_segmentation(Eulers, key=2.):
    # segment the grains
    grain_ids = np.zeros_like(Eulers[0, :, :], dtype='int')
    grain_ids += -1  # mark all pixels as -1 (unlabeled)
    rows, cols = Eulers[0, :, :].shape[0], Eulers[0, :, :].shape[1]
    n_grains = 0
    for j in range(cols):
        for i in range(rows):
            if grain_ids[i, j] >= 0:
                continue  # skip pixel
            # create new grain with the pixel as seed
            n_grains += 1
            # print('segmenting grain %d' % n_grains)
            grain_ids[i, j] = n_grains
            points = [(i, j)]
            while len(points) > 0:
                pixel = points.pop()
                p = Eulers[:, pixel[0], pixel[1]]
                p = np.radians(p)
                # look around this pixel
                east = (pixel[0] - 1, pixel[1])
                north = (pixel[0], pixel[1] - 1)
                west = (pixel[0] + 1, pixel[1])
                south = (pixel[0], pixel[1] + 1)
                northeast = (pixel[0]-1, pixel[1] - 1)
                southeast = (pixel[0] - 1, pixel[1] + 1)
                northwest = (pixel[0] + 1, pixel[1] - 1)
                southwest = (pixel[0]+1, pixel[1] + 1)
                # n1, n2, n3, n4 = (
                #     pixel[0], pixel[1]-2), (pixel[0], pixel[1]+2), (pixel[0]-2, pixel[1]), (pixel[0]+2, pixel[1])
                # m1, m2, m3, m4 = (pixel[0]+1, pixel[1]-2), (pixel[0]+1, pixel[1] +
                #                                             2), (pixel[0]-1, pixel[1]+2), (pixel[0]-1, pixel[1]+2)

                neighbors = [east, north, west, south,
                             northeast, southeast, northwest, southwest]  # n1, n2, n3, n4]#,m1, m2, m3, m4]
                # look at unlabeled connected pixels
                neighbor_list = [n for n in neighbors if
                                 0 <= n[0] < rows and
                                 0 <= n[1] < cols and
                                 grain_ids[n] == -1]
                # print(' * neighbors list is {}'.format([east, north, west, south]))
                for neighbor in neighbor_list:
                    # check misorientation
                    q = Eulers[:, neighbor[0], neighbor[1]]
                    q = np.radians(q)
                    mis = theta(p[0], p[1], p[2], q[0], q[1], q[2])
                    if mis < key:
                        # add to this grain
                        grain_ids[neighbor] = n_grains
                        # add to the list of candidates
                        points.append(neighbor)
    # print('\n%d grains were segmented' % len(np.unique(grain_ids)))
    # grain_ids - labeled grains
    # coloring labeled segments
    kseg_img = color.label2rgb(grain_ids, bg_label=0)
    return kseg_img, grain_ids, len(np.unique(grain_ids))


def grod(labels, ngrains, Eulers):
    grod_map = np.empty_like(Eulers[0, :, :])
    prop = regionprops(labels)
    for i in range(ngrains):
        arr1 = prop[i].image
        offx, offy, _, _ = (prop[i].bbox)
        gx, gy = np.where(arr1 == True)
        gx, gy = gx+offx, gy+offy
        mean0 = np.mean(Eulers[0, gx, gx])
        mean1 = np.mean(Eulers[1, gx, gy])
        mean2 = np.mean(Eulers[2, gx, gy])
        mean0, mean1, mean2 = np.radians(
            mean0), np.radians(mean1), np.radians(mean2)
        h = np.stack([gx, gy])
        # e1,e2,e3 = Eulers[0,gx,gy], Eulers[1,gx,gy], Eulers[2,gx,gy]
        e1, e2, e3 = np.radians(Eulers[0, :, :]), np.radians(
            Eulers[1, :, :]), np.radians(Eulers[2, :, :])
        for j in range(len(gx)):
            p, q = gx[j], gy[j]
            grod_map[p, q] = theta(mean0, mean1, mean2,
                                   e1[p, q], e2[p, q], e3[p, q])

    return grod_map


def get_color_IPF_legend(uvw, issorted=False):
    if isinstance(uvw, (list, tuple)):
        uvw = np.array(uvw)
    shape = uvw.shape
    ndim = uvw.ndim
    if not issorted:
        uvw = np.abs(uvw)
        uvw = np.sort(uvw)
        # Select uvw where w >= u >= v
        uvw = uvw[:, np.array([1, 0, 2])]

    if ndim != 2:
        uvw = uvw.reshape(-1, 3)
    R = uvw[:, 2] - uvw[:, 0]
    G = uvw[:, 0] - uvw[:, 1]
    B = uvw[:, 1]
    # By default, whitespot is in the barycenter of the unit triangle
    whitespot = np.array([0.48846011, 0.22903335, 0.84199195])
    pwr = .75

    # Select uvw where w >= u >= v
    whitespot = np.sort(whitespot)
    whitespot = whitespot[[1, 0, 2]]

    kR = whitespot[2] - whitespot[0]
    kG = whitespot[0] - whitespot[1]
    kB = whitespot[1]

    R = (R/kR)**pwr
    G = (G/kG)**pwr
    B = (B/kB)**pwr

    rgb = np.array([R, G, B])
    rgbmax = np.max(rgb, axis=0)
    # normalize rgb from 0 to 1 and then from 0 to 255
    rgb = rgb*255/rgbmax

    rgb = rgb.astype(np.uint8).T

    if ndim != 2:
        rgb = rgb.reshape(shape)
    return rgb


def ipf_legend(ax=None, n=512):
    xmax, ymax = 2.**.5 - 1, (3.**.5 - 1)/2.
    dx = xmax/(n-1)
    dy = ymax/(n-1)

    # map n x n square around unit triangle
    xp, yp = np.meshgrid(np.linspace(0, xmax, n), np.linspace(0, ymax, n))
    xp, yp = xp.ravel(), yp.ravel()
    # convert projected coordinates (xp, yp) to uvw directions
    u, v, w = 2*xp, 2*yp, 1-xp**2-yp**2
    uvw = np.vstack([u, v, w]).T

    color = np.ndarray(uvw.shape)
    # select directions that will fit inside the unit triangle, i.e.,
    # only those where w >= u >= v
    sel = (w >= u) & (u >= v)
    # uvw directions to corresponding color
    color[sel] = get_color_IPF_legend(uvw[sel])
    # fill points outside the unit triangle in white
    color[~sel] = [255, 255, 255]

    # img_pil = toimage(color.reshape(n, n, 3))
    if ax is None:
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)

    ax.set_aspect('equal')
    ax.axis('off')
    im = color.reshape(n, n, 3)
    im = np.uint8(im)
    ax.imshow(im, origin='lower')

    ax.annotate('001', xy=(0, 0), xytext=(0, -10),
                textcoords='offset points', ha='center', va='top', size=8)
    ax.annotate('101', xy=(n-2, 0), xytext=(0, -10),
                textcoords='offset points', ha='center', va='top', size=8)
    ax.annotate('111', xy=(0.87*n, n-2), xytext=(0, 10),
                textcoords='offset points', ha='center', va='bottom', size=8)
