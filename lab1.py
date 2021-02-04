from pathlib import Path
import os

import numpy as np
import PIL.Image


try:
    IMAGE_DIRECTORY = Path(os.environ['CVL_LAB_IMAGEDIR'])
except KeyError:
    IMAGE_DIRECTORY = Path('F:/kurser/master/tsbb15/Lab1/images')

if not IMAGE_DIRECTORY.exists():
    raise RuntimeError("Image directory '{}' does not exist. Try setting the CVL_LAB_IMAGEDIR environment variable".format(IMAGE_DIRECTORY))


def load_lab_image(filename):
    """Load a grayscale image by filename from the CVL image directory
    
    Example:
    >>> img = load_lab_image('cornertest.png')    
    """
    path = str(IMAGE_DIRECTORY / filename)
    return np.asarray(PIL.Image.open(path).convert('L'))


def get_cameraman():
    "Return I, J and true (col, row) displacement"
    n = 10 # Border crop
    img = load_lab_image('cameraman.tif')
    I = img[n:-n, n:-n]
    x, y = 1, -2
    J = img[n-y:-n-y, n-x:-n-x]
    assert I.shape == J.shape
    return I, J, (x, y)
