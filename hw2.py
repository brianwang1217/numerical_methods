import numpy as np
import random
import math as m
import matplotlib.pyplot as plt
import matplotlib

''' Generating a random vector '''
arr = random.sample(range(101), 50)
x = np.array(arr)

''' Calculating pi using Monte Carlo '''
def calculate_pi(x, y):
    total = 0
    count = 0
    for i in range(len(x)):
        if m.sqrt(y[i] * y[i] + x[i] * x[i]) <= 1.0:
            count += 1
        total += 1
    return (float(count)/total) * 4

pi = np.zeros(7)
for i in range(0, 7):
    pi[i] = calculate_pi(xs[:10**i], ys[:10**i])

plt.plot(np.array(np.log([1, 10, 100, 10**3, 10**4, 10**5, 10**6])), np.log(pi) - np.log(m.pi))
plt.xlabel('number of coordinates')
plt.ylabel('pi estimation')
plt.title('estimation of pi over number of coordinates')

''' The Birthday Problem '''
np.random.seed(Seed)
## DO NOT REMOVE THE ABOVE LINE, OR CHANGE THE 'Seed' VARIABLE

def duplicate_birthdays(n):
    # Generate 1000 simulations of rooms (with the birthdays of n people in each room)
    # Compute the number of rooms with duplicate birthdays.
    # Returns the number of rooms with duplicate birthdays
    simulation = np.zeros([1000,n])
    for i in range(0, 1000):
        simulation[i] = genroom(n)

    duplicates = 0
    for room in simulation:
        if len(room) != len(np.unique(room)):
            duplicates += 1
    return duplicates

prob_n = np.zeros(99)
# Part 1
for n in range(2, 101):
    # call function duplicate_birthdays(n)
    # update the array prob_n
    duplicates = duplicate_birthdays(n)
    prob_n[n - 2] = float(duplicates) / 1000;

# Part 2
# Estimate perc_50
perc_50 = 0
for i in range(0, len(prob_n)):
    if prob_n[i] > 0.5:
        perc_50 = i + 2
        break

# Part 3
# Plot prob_n
plt.plot(np.arange(99) + 2, prob_n)
plt.xlabel('n birthdays')
plt.ylabel('probability of duplicate birthdays')
plt.title('Probability of duplicate birthdays over n birthdays.')
