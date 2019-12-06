import numpy as np
import random
import math
from scipy.signal import convolve2d

''' Page 3 '''
def dfunc(x,dx):
    # ADD CODE HERE
    df_fd = np.zeros(len(x))
    for i in range(len(df_fd)):
        copy = x.copy()#curr = x[i]
        copy[i] = x[i] + dx
        df_fd[i] = (func(copy) - func(x)) / (dx)
#        x[i] = curr
    print(df_fd)
    return df_fd

print(df)

''' Page 4 '''
def prepare_filter(rmin):
    # ADD CODE HERE
    H = np.zeros((2 * math.floor(rmin) + 1, 2 * math.floor(rmin) + 1))
    for i in range(len(H)):
        for j in range(len(H[0])):
            H[i][j] = max(0, rmin - math.sqrt((i + 0.5 - len(H) / 2.0) ** 2 + (j + 0.5 - len(H[0]) / 2.0) ** 2))
    print(H)
    return H

''' Page 5 '''
def prepare_filter(rmin):
    H = np.zeros((2 * math.floor(rmin) + 1, 2 * math.floor(rmin) + 1))
    for i in range(len(H)):
        for j in range(len(H[0])):
            H[i][j] = max(0, rmin - math.sqrt((i + 0.5 - len(H) / 2.0) ** 2 + (j + 0.5 - len(H[0]) / 2.0) ** 2))
    hs = convolve2d(H ,np.ones((nelx, nely)))[int(len(H) / 2):int(-len(H) / 2), int(len(H[0]) / 2):int(-len(H[0]) / 2)]
    print(np.shape(hs))
    return H, hs

def filter_design_variable(x,H,hs):
    x = x.reshape(nelx, nely)
    xhat = convolve2d(x, H, mode='same')
    xhat /= hs
    xhat = np.matrix.flatten(xhat)
    print(xhat)
    print(x1f)
    return xhat
