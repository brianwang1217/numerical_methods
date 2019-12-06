import numpy as np
import numpy.linalg as la

''' Secant '''
roots = np.zeros(5)
new_xks = []
new_xks.append(xks[0])
new_xks.append(xks[1])
for i in range(5):
    roots[i] = (new_xks[-2] * f(new_xks[-1]) - new_xks[-1] * f(new_xks[-2])) / (f(new_xks[-1]) - f(new_xks[-2]))
    new_xks.append(roots[i])
print(roots)

print(xks.shape)

''' Bisection '''
zeros = []
for tuple in brackets:
    p = tuple[0]
    q = tuple[1]
    m = (p + q) / 2
    past_iter = True
    i = 0
    for i in range(n_iter):
        if (np.sign(function(p)) == np.sign(function(q)) or p >= q):
            break
        if abs(function(m)) <= epsilon:
            zeros.append(m)
            past_iter = False
            break
        else:
            if (np.sign(function(p)) == np.sign(function(m))):
                p = m
                m = (m + q) / 2
            elif (np.sign(function(q)) == np.sign(function(m))):
                q = m
                m = (p + m) / 2
    if past_iter:
        zeros.append(None)
print(zeros)

''' Newton '''
def f(x,y):
    return np.array([x**3 - y**2, x+y*x**2 - 2])

# A function that returns the Jacobian may be useful
def J(x,y):
    return np.array([[3 * x ** 2, -2 * y], [1 + 2 * x * y, x ** 2]])

root = np.zeros(2)
root[0] = xi[0]
root[1] = xi[1]
while la.norm(f(root[0], root[1]), 2) >= tol:
    root = root - la.solve(J(root[0], root[1]), f(root[0], root[1]))

res = la.norm(f(root[0], root[1]), 2)
