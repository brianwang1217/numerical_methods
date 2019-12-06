import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

''' Line Fit '''
y = np.array([12, 41, 63, 72, 78, 80, 83, 88, 84, 90])
x = np.array([2005, 2006, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015])

A = np.vstack([x, np.ones(len(x))]).T
c1, c0 = np.linalg.lstsq(A, y)[0]

plt.scatter(x, y)
yfunct = c1 * x + c0
plt.plot(x, yfunct)
plt.xlabel("year")
plt.ylabel("percent")
plt.title("percent of teens using social media per year")
plt.show()

''' Quadratic Fit '''
x = np.array([2005, 2006, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015])
y = np.array([12, 41, 63, 72, 78, 80, 83, 88, 84, 90])
A = np.polyfit(x, y, 2)

c2 = A[0]
c1 = A[1]
c0 = A[2]

plt.scatter(x, y)
plt.xlabel("year")
plt.ylabel("percent")
yplt = x ** 2 * c2 + x * c1 + c0
plt.title("percent of teens using social media per year")
plt.plot(x, yplt)
plt.show()

''' Exponential Fit '''
#V = np.zeros(some_shape)
#V[:,0] = # ...
V = np.vstack([x, np.ones(len(x))]).T
newY = np.log(y)
c1, c0 = np.linalg.lstsq(V, newY)[0]
coeffs = np.array( [np.e ** c0, c1])

plt.plot(xp, coeffs[0]*np.exp(coeffs[1]*xp))
plt.plot(x, y, "o")
