import numpy as np
import matplotlib.pyplot as plt

''' Trigonometric Interpolation '''
b = np.linspace(0, 7, 7)
b = func(nodes)

A = np.ones((7, 7))
A[1] *= np.sin(nodes)
A[2] *= np.cos(nodes)
A[3] *= np.sin(nodes * 2)
A[4] *= np.cos(nodes * 2)
A[5] *= np.sin(nodes * 3)
A[6] *= np.cos(nodes * 3)
A = A.T
print(A)

coeffs = np.linalg.solve(A, b)

x = np.linspace(0, np.pi * 2, 100)
plt.title('interpolant, function, and data points')
plt.xlabel('x values')
plt.ylabel('function')
plt.plot(x, func(x))
plt.plot(nodes, func(nodes), 'ro', markersize=5)
plt.legend(['actual function', 'points'])
plt.show()

''' Polynomial Interpolation '''
b3 = np.linspace(-1, 1, 4)
b5 = np.linspace(-1, 1, 6)
b15 = np.linspace(-1, 1, 16)

A3= np.ones((4, 4))
A5 = np.ones((6, 6))
A15 = np.ones((16, 16))

for i in range(4):
    A3[i] *= b3 ** i

for i in range(6):
    A5[i] *= b5 ** i

for i in range(16):
    A15[i] *= b15 ** i

b3 = 1 / (1 + 25.0 * b3 ** 2)
b5 = 1 / (1 + 25.0 * b5 ** 2)
b15 = 1 / (1 + 25.0 * b15 ** 2)

A3 = A3.T
A5 = A5.T
A15 = A15.T

coefficients_3 = np.linalg.solve(A3, b3)
coefficients_5 = np.linalg.solve(A5, b5)
coefficients_15 = np.linalg.solve(A15, b15)

x3 = np.linspace(-1, 1, 4)
x5 = np.linspace(-1, 1, 6)
x15 = np.linspace(-1, 1, 16)
x = np.linspace(-1, 1, 50)
vals_3 = np.polyval(np.flip(coefficients_3, axis=0), x)
vals_5 = np.polyval(np.flip(coefficients_5, axis=0), x)
vals_15 = np.polyval(np.flip(coefficients_15, axis=0), x)

plt.title('interpolation with coefficients')
plt.ylabel('interpolatino')
plt.xlabel('xlabels')
plt.plot(x, vals_3)
plt.plot(x, vals_5)
plt.plot(x, vals_15)
plt.plot(x, 1 / (1 + 25.0 * x ** 2))
plt.legend(['3', '5', '15', 'actual'])
fig = plt.gca()

''' Flights '''
A1 = np.ones((3, len(prices)))

A1[1] = times
A1[0] = times ** 2

A1 = A1.T
p1 = np.linalg.lstsq(A1, prices)[0]

derivative_values = np.array(2 * p1[0] * times + p1[1])
print(derivative_values)

# we have derivative values, now find 2nd derivative and the values closest to 0
second_derivative = np.array(2 * p1[0] * np.ones(len(times)))

while True:
    min = derivative_values[0]
    index = 0
    for i in range(len(derivative_values)):
        if abs(derivative_values[i]) < abs(min):
            index - i
            min = derivative_values[i]
    if second_derivative[index] > 0:
        best_time = times[index]
        break
    else:
        np.delete(derivative_values, index)
        np.delete(second_derivative, index)

A2 = np.vstack([times, np.ones(len(times))]).T
p2 = np.linalg.lstsq(A2, prices)[0]

A3 = np.vstack([times ** 5, times ** 4, times ** 3, times ** 2, times, np.ones(len(times))]).T
p3 = np.linalg.lstsq(A3, prices)[0]

plt.plot(times, prices)
plt.xlabel('times')
plt.ylabel('prices')
plt.title('price over time')
plt.legend(['p1', 'p2', 'p3'])
