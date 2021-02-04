import cv2
import numpy as np
import scipy
from scipy.signal import convolve2d as conv2
import matplotlib.image as mpimage

from matplotlib import pyplot as plt
import lab1

from regularizedderivatives import regularized_values
from estimateD import estimate_d
from interpolate import interpolate
from orientationTensor import orientation_tensor

I, J, dTrue = lab1.get_cameraman()

dtot = estimate_d(I, J, 120, 85, (40, 70), max_iterations=10 , min_error=0)

print('dTrue = ', dTrue)
print('dtot = ', (dtot[0][0],dtot[1][0]))

T_field = orientation_tensor(I, 5, 0.5)
plt.figure(1)
plt.imshow(T_field[:,:,1])
plt.figure(2)
plt.imshow(T_field[:,:,2])
plt.figure(3)
plt.imshow(T_field[:,:,0])
plt.show()