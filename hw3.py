import math
import numpy as np
import matplotlib.pyplot as plt

# Score: 13.33/20
''' Approximation of an infinite series '''
err_0_1, err0_3, err_pi4_3 = 0, 0, 0
x_0, x_pi = 0, math.pi / 4
val_1, val_3, val_pi3 = 0, 0, 0

for i in range(1, 3):
    # for 0_1
    if (i <= 1):
        val_1 += (-1) ** (i + 1) * ((0.7 - x_0) ** (2 * i - 1)) / math.factorial(2 * i - 1)

    # for 0_3
    val_3 += (-1) ** (i + 1) * ((0.7 - x_0) ** (2 * i - 1)) / math.factorial(2 * i - 1)
    # for pi/4_3
    val_pi3 += (-1) ** (i + 1) * ((0.7 - x_pi) ** (2 * i - 1)) / math.factorial(2 * i - 1)

err_0_1 = abs(val_1 - math.sin(0.7 - x_0)) / math.sin(0.7 - x_0)
err_0_3 = abs(val_3 - math.sin(0.7 - x_0)) / math.sin(0.7 - x_0)
err_pi4_3 = abs(val_pi3 - math.sin(0.7 - x_pi)) / math.sin(0.7 - x_pi)

''' Approximation of an infinite series 2 '''
exp_approx = np.zeros(10)
abs_error = np.zeros(10)
rel_error = np.zeros(10)
N = 0

for i in range(10):
    for j in range(i + 1):
        exp_approx[i] += x ** j / math.factorial(j)

plt.plot(np.arange(10), exp_approx)
plt.title("approximate exp values with increasing n using taylor series")
plt.ylabel("approx")
plt.xlabel("n value")

for i in range(10):
    abs_error[i] = abs(exp_approx[i] - math.e ** x)
    rel_error[i] = abs_error[i] / math.e ** x

for i in range(9, -1, -1):
    if rel_error[i] <= desired_rel_error / 100:
        N = i
