import scipy
import numpy as np

def interpolate(in_im, y, x, height, width):
    # creates interpolated object for in_im over area x:x+width, y:y+height
    xcoords = np.arange(x, x+width)
    ycoords = np.arange(y, y+height)

    interpol_im = scipy.interpolate.RectBivariateSpline(ycoords,xcoords,in_im[y:y+height,x:x+width])

    return interpol_im

