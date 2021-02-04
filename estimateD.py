import cv2
import numpy as np
import scipy
from scipy.signal import convolve2d as conv2
import matplotlib.image as mpimage

from matplotlib import pyplot as plt
import lab1

from regularizedderivatives import regularized_values
from estimateT import estimate_T
from estimateE import estimate_E
from interpolate import interpolate


def estimate_d(I, J, x, y, window_size, max_iterations=50, min_error=0):
    dtot = np.zeros((2, 1))
    im_width = np.shape(J)[1]
    im_height = np.shape(J)[0]
    xcoords = np.arange(0, im_width)
    ycoords = np.arange(0, im_height)

    Ig, Jg, Jgdx, Jgdy = regularized_values(I, J, 5, 1)

    Jg_og = Jg
    Jgdx_og = Jgdx
    Jgdy_og = Jgdy

    for tmp in range(max_iterations):

        T = estimate_T(Jgdx, Jgdy, x, y, window_size)
        e = estimate_E(Ig, Jg, Jgdx, Jgdy, x, y, window_size)
        d = np.linalg.solve(T, e)
        dtot += d
        if np.linalg.norm(d) < min_error:
            break

        Jg_interpol = interpolate(Jg_og, 0, 0, im_height, im_width)
        Jg = Jg_interpol(np.arange(dtot[1], im_height + dtot[1]), np.arange(dtot[0], im_width + dtot[0]), grid=True)

        Jgdx_interpol = interpolate(Jgdx_og, 0, 0, im_height, im_width)
        Jgdx = Jgdx_interpol(np.arange(dtot[1], im_height + dtot[1]), np.arange(dtot[0], im_width + dtot[0]), grid=True)

        Jgdy_interpol = interpolate(Jgdy_og, 0, 0, im_height, im_width)
        Jgdy = Jgdy_interpol(np.arange(dtot[1], im_height + dtot[1]), np.arange(dtot[0], im_width + dtot[0]), grid=True)
    return dtot
