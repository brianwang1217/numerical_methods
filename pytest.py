import numpy as np
import numpy.linalg as la
import scipy.optimize as opt

import numpy as np

def genRandomVect():
    # define return numpy array
    x = np.array([[7.07, 13.22], [13.22, 60.93]]);
    print(np.linalg.eig(x)[0]);
genRandomVect();

# def f(x1, x2):
#     return np.array([3 * x1 * x2 + 5, x1 ** 3 + x2 ** 2 + 3])
#
# def df(x1, x2):
#     return np.array([[3 * x2, 3 * x1], [3 * x1 ** 2, 2 * x2]])
#
# x = np.array([-2, 0])
#
# x = x - np.linalg.solve(df(x[0], x[1]), f(x[0], x[1]))
#
# print(df(-1, 2))
# print(x)


''' p2'''
# def f(X):
#     x = X[0]
#     y = X[1]
#     return 11 * x ** 2 - 6 * x + 7 + 25 * y ** 2 - x * y
#
# def df(X):
#     x = X[0]
#     y = X[1]
#     return np.array([22 * x - 6 - y, 50 * y - x])
#
# def ddf(X):
#     x = X[0]
#     y = X[1]
#     return np.array([[22, -1], [-1, 50]])
#
# count = 0
# X = [1, 2]
# while (la.norm(X - la.solve(ddf(X), df(X)), 2) > 10 ** -7):
#     X = X - la.solve(ddf(X), df(X))
#     count += 1
# print(X)

'''Jacobian for Nonlinear Least-Squares Residual'''
# t = np.array([-5, -2, 1, 2])
# y = np.array([-2, -10, -8, 6])
# a = .25
# b = np.pi
#
# # the function is just a * np.cos(b * t) here
# residual = y - a * np.cos(b * t)
# print(residual)
# for i in range(4):
#     print(np.sign(np.cos(b * t[i])))

'''function for r_newton for optimization'''

# def func(x, y):
#     return (0.5 * x ** 2 + 2*x*y + 2.5 * y ** 2)
#
# def df(x, y):
#     return np.array([x + 2*y, 2*x + 5*y])
#
# def H(x, y):
#     return np.array([[1, 2], [2, 5]])
#
# # initial values of r_newton
# r_newton = np.array([3, 2])
# iteration_count_newton = 0
# delta = abs(func(r_newton[0], r_newton[1]))
# print(np.linalg.inv(H(r_newton[0], r_newton[1])) @ df(r_newton[0], r_newton[1]))
# while np.linalg.norm(delta) > 10 ** -5:
#     iteration_count_newton += 1
#     r_newton = r_newton - np.linalg.inv(H(r_newton[0], r_newton[1])) @ df(r_newton[0], r_newton[1])
#     delta = abs(df(r_newton[0], r_newton[1]))
# print(iteration_count_newton)

''' calculate s_0 (from x_k+1 = x_k + x_k) '''
# df = np.array([35, 28.5464871])
# h = np.array([[213, 35], [35, 19.8385316]])
# r_newton = np.array([0, 1])
#
# print(r_newton - np.linalg.inv(h) @ df)

''' rank 1 approximation '''
# Ar = np.zeros((len(U), len(Vt)))
# # for i in range(1):
# Ar += 7 * np.outer(U.T[0], Vt[0])
# print(Ar)
# Ar += 5 * np.outer(U.T[1], Vt[1])
# print(Ar)
# # entire matrix
# for i in xrange(r):
#     Ar += s[i] * np.outer(u.T[i], v[i])
