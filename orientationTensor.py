import numpy as np
from scipy.signal import convolve2d as conv2


def orientation_tensor(img, grad_ksize, grad_sigma, ksize, sigma):
    lp = np.atleast_2d(np.exp(-0.5 * (np.arange(-ksize, ksize + 1) / sigma) ** 2))
    lp = lp / np.sum(lp)  # normalize the filter
    df = np.atleast_2d(-1.0 / np.square(sigma) * np.arange(-ksize, ksize + 1) * lp) # 1d deriv filt

    dx = conv2(img, df, mode = "same")
    dy = conv2(img, df.T, mode = "same")

    T_field = np.empty(np.shape(img))

    return T_field