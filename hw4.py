import numpy as np
import math
import random

''' Floating Point Playground'''
k = -1
while True:
    if a != (a + 10 ** k):
        k -= 1
    else:
        break
k += 1

''' Smallest Number Python '''
smallest_num = 1.0
while smallest_num > 0.0:
    if smallest_num / 2.0 > 0:
        smallest_num = smallest_num / 2.0
    else:
        break

''' Floating Point Summation '''
data_sum = 0
sorted = np.sort(data)

for point in sorted:
    data_sum += point

''' Standard Deviation and Cancellation '''
sequence = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
var_seq_tp = 0.0
var_seq_op = 0.0

for data in sequence:
    var_seq_tp += (data - np.mean(sequence)) ** 2
    var_seq_op += (data ** 2)

var_seq_tp = (var_seq_tp / (len(sequence) - 1)) ** 0.5
var_seq_op = var_seq_op - len(sequence) * (np.mean(sequence) ** 2)
var_seq_op = (var_seq_op / (len(sequence) - 1)) ** 0.5

bad_sequence = np.array([1.00000000000000000000253, 1.000000000000000614, 1.000000000000000000361, 1.000000000000000000888, 1.000000000000000911, 1.0000000000000000000111,  1.00000000000000000014, 1.00000000000000000654, 1.000000000000000000734, 1.00000000000000000241])
