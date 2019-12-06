import matplotlib.pyplot as pt
import numpy as np
import math
# Score: 12/18
''' SVD and PCA '''
# a) Plot X
mean = np.zeros(2,)
n = len(X[0])
a = 0
b = 0
for i in range(n):
    a += X[0][i]
    b += X[1][i]
mean[0] = a / n
mean[1] = b / n
# b) Plot this on top
X[0] = (X[0] - mean[0]) * (1/math.sqrt(n - 1))
X[1] = (X[1] - mean[1]) * (1/math.sqrt(n - 1))

u, s, vh = np.linalg.svd(X, full_matrices=True)

shape = (2,2)
sHat = np.zeros(shape)
sHat[0][0] = s[0]
sHat[1][1] = s[1]

principal_components = np.transpose(u @ sHat)
# c) Y

# d) Compute X_prime and plot as before
