import numpy as np

def harris(kappa, T_field):
    # takes in T_field as a img where each pixel has T11, T12, T22 respectively in 3rd dim

    H_field = np.zeros((np.shape(T_field)[0], np.shape(T_field)[1]))
    for row in range(np.shape(T_field)[0]):
        for col in range(np.shape(T_field)[1]):
            T11 = T_field[row, col, 0]
            T12 = T_field[row, col, 1]
            T22 = T_field[row, col, 2]
            H_field[row, col] = T11 * T22 - T12 * T12 - kappa * (T11 + T22)**2

    return H_field