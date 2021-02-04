import numpy as np
from scipy.signal import convolve2d as conv2


def orientation_tensor(img, gradksize, gradsigma, ksize, sigma):
    # returns img with tensors in 3rd dim, T11, T12, T22 respectively

    lp = np.atleast_2d(np.exp(-0.5 * (np.arange(-ksize, ksize + 1) / sigma) ** 2))
    lp = lp / np.sum(lp)  # normalize the filter
    df = np.atleast_2d(-1.0 / np.square(sigma) * np.arange(-ksize, ksize + 1) * lp) # 1d deriv filt

    dx = conv2(img, df, mode = "same")
    dy = conv2(img, df.T, mode = "same")

    T_field = np.zeros((np.shape(img)[0], np.shape(img)[1], 3))
    T_field[:, :, 0] = dx * dx
    T_field[:, :, 1] = dx * dy
    T_field[:, :, 2] = dy * dy

    # low pass filter
    lp = np.atleast_2d(np.exp(-0.5 * (np.arange(-gradksize, gradksize + 1) / gradsigma) ** 2))
    lp = lp / np.sum(lp)  # normalize the filter

    T_field[:, :, 0] = conv2(conv2(T_field[:, :, 0], lp, mode='same'), lp.T, mode='same')
    T_field[:, :, 1] = conv2(conv2(T_field[:, :, 1], lp, mode='same'), lp.T, mode='same')
    T_field[:, :, 2] = conv2(conv2(T_field[:, :, 2], lp, mode='same'), lp.T, mode='same')

    return T_field