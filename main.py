import cv2
import numpy as np
import scipy
from scipy.signal import convolve2d as conv2
import matplotlib.image as mpimage
from matplotlib.patches import Circle
from matplotlib import pyplot as plt
import lab1

from regularizedderivatives import regularized_values
from estimateD import estimate_d
from interpolate import interpolate
from orientationTensor import orientation_tensor
from harris import harris



I, J, dTrue = lab1.get_cameraman()

dtot = estimate_d(I, J, 120, 85, (40, 70), max_iterations=5, min_error=0)

print('dTrue = ', dTrue)
print('dtot = ', (dtot[0][0], dtot[1][0]))

###############

testim = cv2.imread('F:/kurser/master/tsbb15/CE/TSBB15/images/chessboard/img1.png', cv2.IMREAD_GRAYSCALE)

T_field = orientation_tensor(testim, 10, 1, 10, 1)

plt.figure(1)
plt.imshow(T_field[:, :, 1])
plt.figure(2)
plt.imshow(T_field[:, :, 2])
plt.figure(3)
plt.imshow(T_field[:, :, 0])

kappa = 0.05
H_field = harris(kappa, T_field)
plt.figure(4)
plt.imshow(H_field, cmap='gray')

threshold = np.amax(H_field) * 0.7
H_mask = H_field > threshold
H_thresh = H_field * H_mask
plt.figure(5)
plt.imshow(H_thresh, cmap='gray')

img_max = scipy.ndimage.filters.maximum_filter(H_thresh, size=3)
[row, col] = np.nonzero((H_thresh == img_max) * H_mask)  # remove dilation

# save the K best ones
K = 5
feats = np.zeros((len(row), 3))
for i in range(len(row)):
    feats[i, 0] = H_thresh[row[i], col[i]]
    feats[i, 1] = row[i]
    feats[i, 2] = col[i]
feats_sorted = feats[feats[:, 0].argsort()]

# get K feat coords
feats_coords = np.zeros((K, 2))
for i in range(K):
    end = np.shape(feats_sorted)[0] - 1
    feats_coords[i, 0] = feats_sorted[end - i, 1]
    feats_coords[i, 1] = feats_sorted[end - i, 2]

plt.close('all')

I = cv2.imread('F:/kurser/master/tsbb15/CE/TSBB15/images/chessboard/img1.png', cv2.IMREAD_GRAYSCALE)

it = 10
_, ax = plt.subplots(1, num="1")
ax.imshow(I, cmap='gray')
for i in range(K): # draw circles
    ax.add_patch(Circle((feats_coords[i, 1], feats_coords[i, 0]), 10, fill=False, edgecolor='red', linewidth=1))

window = 30

for im in range(2, 11):
    print('Processing image ', im)
    J = cv2.imread("F:/kurser/master/tsbb15/CE/TSBB15/images/chessboard/img%d.png" % im, cv2.IMREAD_GRAYSCALE)
    _, ax = plt.subplots(1, num="%d" % im)
    ax.imshow(J, cmap='gray')
    for i in range(K):
        x = feats_coords[i, 1] # prev image points
        y = feats_coords[i, 0]
        d = estimate_d(I, J, x, y, (window, window), it, 0.1)
        feats_coords[i, 1] += d[0] # pos update
        feats_coords[i, 0] += d[1]
        print(feats_coords[i])
        ax.add_patch(Circle((feats_coords[i, 1], feats_coords[i, 0]), 10, fill=False, edgecolor='red', linewidth=1))
    I = J

plt.show()
