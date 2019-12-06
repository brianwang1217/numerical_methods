import scipy
import numpy as np
import matplotlib.pyplot as plt

''' Drug Metabolism '''
M = np.array([[.7, 0, 0, 0], [.1, .5, .1, 0], [.1, .4, .8, 0], [.1, .1, .1, 1]])
arr = np.array([1, 0, 0, 0])
hours = 0
while (arr[3] < .95):
    arr = M @ arr
    hours += 1

''' Train Stations '''
mat = []
A = [0, 1, 1, 1, 0]
B = [1, 0, 0, 1, 0]
C = [1, 1, 0, 1, 0]
D = [1, 0, 0, 0, 1]
E = [1, 0, 1, 0, 0]
mat.append(A)
mat.append(B)
mat.append(C)
mat.append(D)
mat.append(E)

mat = np.array(mat)
mat = mat / mat.sum(axis=0, keepdims=1)

x0 = [1, 0, 0, 0, 0]
its = 20

for k in range(its):
    x0 = mat @ x0

print(x0)
prob = np.array(x0)
print(prob)

''' Eigen Power Iteration '''
cnt = np.zeros(n)
shape = (n, 2)
eigenval1 = np.zeros((n,))
eigenvec1 = np.zeros(shape)
eigenval2 = np.zeros((n,))
eigenvec2 = np.zeros(shape)

for i in range(n):
    # power iteration
    count = 1
    xk = np.array([1/(2 ** 0.5), 1/(2 ** 0.5)])

    while np.linalg.norm((As[i] @ xk / np.linalg.norm(As[i] @ xk, 2)) - xk, 2) > (10 ** -12):
        xk = As[i] @ xk
        xk /= np.linalg.norm(xk, 2)
        count += 1
    eigenvec1[i] = xk
    eigenval1[i] = (xk.T @ As[i] @ xk) / (xk.T @ xk)
    cnt[i] = count

    # inverse power iteration
    p, l, u = scipy.linalg.lu(As[i])
    xl = np.array([1/(2 ** 0.5), 1/(2 ** 0.5)])
    while np.linalg.norm((As[i] @ xl / np.linalg.norm(As[i] @ xl, 2)) - xl, 2) > (10 ** -12):
        y = scipy.linalg.solve_triangular(l, np.dot(p.T, xl), lower=True)
        xl = scipy.linalg.solve_triangular(u, y)
        xl = xl / np.linalg.norm(xl, 2)
    eigenvec2[i] = xl
    eigenval2[i] = (xl.T @ As[i] @ xl) / (xl.T @ xl)

plt.plot(cnt, eigenval1/eigenval2)
plt.xlabel("count to 10**-12 error")
plt.ylabel("eig1 / eig2")
plt.title("count to eig1/eig2")
plt.show()
