import math
import numpy as np


def estimate_E(Ig, Jg, Jgdx, Jgdy, x, y, window_size):
    e = np.zeros((2,1))

    # define the window
    x0 = math.floor(x - window_size[1] / 2)
    y0 = math.floor(y - window_size[0] / 2)
    x1 = x0 + window_size[1]
    y1 = y0 + window_size[0]

    Ig_win = Ig[y0:y1, x0:x1]
    Jg_win = Jg[y0:y1, x0:x1]
    Jgdx_win = Jgdx[y0:y1, x0:x1]
    Jgdy_win = Jgdy[y0:y1, x0:x1]

    e[0] = np.sum(np.multiply(Ig_win-Jg_win, Jgdx_win))
    e[1] = np.sum(np.multiply(Ig_win-Jg_win, Jgdy_win))
    return e