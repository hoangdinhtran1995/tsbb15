import numpy as np
from scipy.signal import convolve2d as conv2

# Takes images I and J and lp filters them with gaussian of size ksize x ksize and sigma
# Also computs gradients Jgdx and Jgdy of img J

def regularized_values(I, J, ksize, sigma):
    # gets the gradient images
    lp = np.atleast_2d(np.exp(-0.5 * (np.arange(-ksize, ksize + 1) / sigma) ** 2))
    lp = lp / np.sum(lp)  # normalize the filter
    df = np.atleast_2d(-1.0 / np.square(sigma) * np.arange(-ksize, ksize + 1) * lp) # 1d deriv filt
    Ig = conv2(conv2(I, lp, mode="same"), lp.T, mode="same")
    Jg = conv2(conv2(J, lp, mode="same"), lp.T, mode="same")
    Jgdx = conv2(J, df, mode = "same")
    Jgdy = conv2(J, df.T, mode = "same")
    return Ig, Jg, Jgdx, Jgdy
