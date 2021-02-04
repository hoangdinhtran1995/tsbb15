import cv2
import numpy as np
import scipy
from matplotlib import pyplot as plt
import lab1
import math


def estimate_T(Jgdx, Jgdy, x, y, window_size):
    T = np.zeros((2, 2))

    # define the window
    x0 = math.floor(x - window_size[1] / 2)
    y0 = math.floor(y - window_size[0] / 2)
    x1 = x0 + window_size[1]
    y1 = y0 + window_size[0]

    dx = Jgdx[y0:y1, x0:x1]
    dy = Jgdy[y0:y1, x0:x1]

    T[0,0] = np.sum(dx*dx)
    T[1,0] = np.sum(dx*dy)
    T[0,1] = T[1,0]
    T[1,1] = np.sum(dy*dy)
    return T

