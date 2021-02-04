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

I, J, dTrue = lab1.get_cameraman()

dtot = estimate_d(I, J, 120, 85, (40, 70), max_iterations=10 , min_error=0)

print('dTrue = ', dTrue)
print('dtot = ', (dtot[0][0],dtot[1][0]))

# plt.figure(1)
# plt.imshow(I)
# plt.figure(2)
# plt.imshow(J)
#
#
# print('dTrue =', dTrue)
#
# Ig, Jg, Jgdx, Jgdy = regularized_values(I, J, 6, 1)
# plt.figure(3)
# plt.imshow(Ig)
# plt.figure(4)
# plt.imshow(Jg)
# plt.figure(5)
# plt.imshow(Jgdx)
# plt.figure(6)
# plt.imshow(Jgdy)



#testim = mpimage.imread('F:/kurser/master/tsbb15/CE/TSBB15/images/boat.png').astype('float32') / 255

#test of rectbivarspline
#y = 300
#x = 300
#width = 20
#height = 30
#testim_interp = interpolate(testim, y, x, height, width)
#
#x2 = np.arange(x, x+width, 0.25)
#y2 = np.arange(y, y+height, 0.25)
#tmp = testim_interp(y2, x2)


#plt.figure(7)
#plt.imshow(tmp, cmap = 'gray')
#plt.figure(8)
#plt.imshow(testim[y:y+height,x:x+width], cmap = 'gray')
#
#plt.show()



