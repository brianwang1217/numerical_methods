import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy.optimize import minimize_scalar
from scipy.optimize import least_squares


''' Golden Section Search '''
brackets = []
gs = (np.sqrt(5) - 1) / 2
m1 = a + (1 - gs) * (b - a)
m2 = a + gs * (b - a)

# Begin your modifications below here
# f(), a, b are given variables
fm1 = f(m1)
fm2 = f(m2)
while abs(a - b) >= 10 ** -5:
    if fm1 > fm2:
        a = m1
        m1 = m2
        fm1 = fm2
        m2 = a + gs * (b-a)
        fm2 = f(m2)
    else:
        b = m2
        m2 = m1
        fm2 = fm1
        m1 = a + (1 - gs) * (b-a)
        fm1 = f(m1)

    brackets.append([a, m1, m2, b])

# End your modifications above here

# Plotting code below, no need to modify
x = np.linspace(-10, 10)
plt.plot(x, f(x))

brackets = np.array(brackets)
for i in range(4):
    plt.plot(brackets[:, i], 3*np.arange(len(brackets)), 'o-')

''' Newton's Method vs Steepest Descent '''
def func(x, y):
    return (3 +((x**2)/8) + ((y**2)/8) - np.sin(x)*np.cos((2**0.5)/2*y))

def df(x, y):
    return np.array([x / 4 - np.cos(x) * np.cos((2 ** 0.5) / 2 * y), y / 4 + np.sin(x) * np.sin((2 ** 0.5) / 2* y) * 2 ** 0.5 / 2])

def H(x, y):
    return np.array([[1/4 + np.sin(x) * np.cos(2 ** (1/2) / 2 * y), (2 ** 0.5) / 2 * np.cos(x) * np.sin((2 ** 0.5) / 2 * y)], [(2 ** 0.5) / 2 * np.cos(x) * np.sin((2 ** 0.5) / 2 * y), 1/4 + 0.5 * np.sin(x) * np.cos((2 ** 0.5) / 2 * y)]])

r_newtons = []
r_sds = []

# NEWTON'S OPTIMIZATION
r_newton = np.array([r_init[0], r_init[1]])
iteration_count_newton = 0
delta = abs(func(r_newton[0], r_newton[1]))
r_newtons.append(r_newton)

while np.linalg.norm(delta) > stop:
    iteration_count_newton += 1
    r_newton = r_newton - np.linalg.inv(H(r_newton[0], r_newton[1])) @ df(r_newton[0], r_newton[1])
    delta = abs(df(r_newton[0], r_newton[1]))
    r_newtons.append(r_newton)


# STEEPEST DESCENT
def obj_func(alpha, x, y, s):
    return (3 +(((x + alpha * s[0])**2)/8) + (((y + alpha * s[1])**2)/8) - np.sin(x + alpha * s[0])*np.cos((2**0.5)/2*(y + alpha * s[1])))

r_sd = np.array([r_init[0], r_init[1]])
iteration_count_sd = 0
delta = abs(func(r_sd[0], r_sd[1]))
old = np.array([0, 0])
r_sds.append(r_sd)

while np.linalg.norm(r_sd - old) > stop:
    old = r_sd
    s = -df(old[0], old[1])
    alpha = minimize_scalar(obj_func, args=(old[0], old[1], s)).x
    r_sd = old + alpha * s
    iteration_count_sd += 1
    r_sds.append(r_sd)

# for some reason we iterate one extra time
r_sds = r_sds[:-1]
r_sd = old
iteration_count_sd -= 1

# calculate log(norm(error))
x_newton = list(range(len(r_newtons)))
x_sd = list(range(len(r_sds)))
y_newton_raw = (r_newtons - r_newton)
y_newton = []
for pair in y_newton_raw:
    y_newton.append(np.log(la.norm(pair)))
y_newton = y_newton[:-1]

y_sd_raw = (r_sds - r_sd)
y_sd = []
for pair in y_sd_raw:
    y_sd.append(np.log(la.norm(pair)))
y_sd = y_sd[:-1]

plt.plot(x_newton[:-1], y_newton)
plt.plot(x_sd[:-1], y_sd)
plt.xlabel('iteration')
plt.ylabel('log of norm of error values')
plt.title('error throughout iterations of newton\'s method and steepest descent')
plt.legend(['newton\'s', 'steepest descent'])
plt.show()

''' Solving Nonlinear Least-Squares '''
def f(x, w):
    return w[0] * np.exp(x * w[1]) + w[2] * np.sin(x * w[3])

# calculates residual
def res(w, x, y):
    func_values = []
    for i in range(len(x)):
        func_values.append(w[0] * np.exp(x[i] * w[1]) + w[2] * np.sin(x[i] * w[3]))
    return y - np.array(func_values)

# jacobian
def jac(w, x, y):
    J = []
    for i in range(len(x)):
        J.append([np.exp(w[1] * x[i]), w[0] * x[i] * np.exp(w[1] * x[i]), np.sin(w[3] * x[i]), w[2] * x[i] * np.cos(w[3] * x[i])])
    return -np.array(J)

r_init = res(w_initial, x_train, y_train)
J_init = jac(w_initial, x_train, y_train)

w = least_squares(res, w_initial, jac, method='lm', args=(x_train, y_train)).x

plt.xlabel('x')
plt.ylabel('y')
plt.title('best fit with nonlinear least-squares')
plt.plot(x_test, f(x_test, w))
plt.scatter(x_test, y_test)
plt.scatter(x_train, y_train)
plt.legend(['plot', 'test', 'train'])
plt.show()
