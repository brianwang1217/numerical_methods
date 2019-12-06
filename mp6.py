import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg as sla

''' Page 2 '''
def lotkaVolterra(x, y):
    #You may want to define this helper function (optional)
    return np.array([8.0 * x - 4.0 * x * y, x * y - 5.0 * y])

def jacobian(x, y):
    J = np.empty((2,2))
    #Complete this function
    return np.array([[8.0 - 4.0 * y, -4.0 * x], [y, x - 5.0]])

def newton(x, y):
    #Complete this function
    root = np.array([x, y])
    while np.linalg.norm(lotkaVolterra(root[0], root[1]), 2) >= 10 ** -10:
        root = root - np.linalg.solve(jacobian(root[0], root[1]), lotkaVolterra(root[0], root[1]))
    return (root[0], root[1])

guesses = ((1.0, 1.0), (96124123123.0, 1.0))

''' Page 3 '''
def lotkaVolterra(x, y):
    # You may wish to implement this helper function
    return np.array([8.0 * x - 4.0 * x * y, x * y - 5.0 * y])

def jacobian(x, y):
    J = np.empty((2,2))
    #Complete this function
    return np.array([[8.0 - 4.0 * y, -4.0 * x], [y, x - 5.0]])

def broyden(x, y):
    #Complete this function
    jac = jacobian(x, y)
    while np.linalg.norm(lotkaVolterra(x, y), 2) >= 10 ** -10:
        root = np.linalg.solve(jac, -1 * lotkaVolterra(x, y))
        delta = lotkaVolterra(x + root[0], y + root[1]) - lotkaVolterra(x, y)
        x = x + root[0]
        y = y + root[1]
        jac = jac + np.outer((delta - jac @ root), root.T) / np.inner(root.T, root)
    return (x, y)

guesses = ((1, 2), (-10, -2))

''' Page 4 '''
def lotkaVolterra(x, y):
    #You may want to define this helper function (optional)
    return np.array([8.0 * x - 4.0 * x * y, x * y - 5.0 * y])

def jacobian(x, y):
    J = np.empty((2,2))
    #Complete this function
    return np.array([[8.0 - 4.0 * y, -4.0 * x], [y, x - 5.0]])

def newton(x, y):
    count = 0
    #Complete this function
    root = np.array([x, y])
    while np.linalg.norm(lotkaVolterra(root[0], root[1]), 2) >= 10 ** -10:
        count += 1
        if (np.linalg.det(jacobian(root[0], root[1])) == 0):
            return -1
        root = root - np.linalg.solve(jacobian(root[0], root[1]), lotkaVolterra(root[0], root[1]))
    #return (root[0], root[1])
    return count

def broyden(x, y):
    #Complete this function
    count = 0
    jac = jacobian(x, y)
    while np.linalg.norm(lotkaVolterra(x, y), 2) >= 10 ** -10:
        count += 1
        if (np.linalg.det(jac) == 0):
            return -1
        root = np.linalg.solve(jac, -1 * lotkaVolterra(x, y))
        delta = lotkaVolterra(x + root[0], y + root[1]) - lotkaVolterra(x, y)
        x = x + root[0]
        y = y + root[1]
        jac = jac + np.outer((delta - jac @ root), root.T) / np.inner(root.T, root)
    #return (x, y)
    return count

shape = (30, 30)
newt_steps = np.zeros(shape)
broyd_steps = np.zeros(shape)

for x in range(30):
    for y in range(30):
        newt_steps[x][y] = newton(x, y)
        broyd_steps[x][y] = broyden(x, y)
