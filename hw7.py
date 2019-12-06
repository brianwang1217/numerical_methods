import numpy as np
import math

''' Estimate Digits '''
digits = 0
x = np.linalg.solve(A, b)
cond = np.linalg.cond(A)
correct_digits = 16 - math.ceil(math.log10(cond))

''' Investigating Conditioning '''
err_xhat = (xtrue - xhat)
rel_err_xhat = np.linalg.norm(err_xhat, 2) / np.linalg.norm(xtrue, 2)
rel_err_Axhat = np.linalg.norm(A @ xtrue - A @ xhat) / np.linalg.norm(A @ xtrue)
cond_A = np.linalg.cond(A, 2)
bound_rel_err_Axhat =  cond_A * rel_err_xhat
print(bound_rel_err_Axhat)

''' Investigating Power Usage '''
A = []
b = []
for val in test_data.values():
    vals = [None] * len(components)
    for tuple in val:
        if (tuple[0] == "PowerConsumed"):
            b.append(tuple[1])
        else:
            vals[components.index(tuple[0])] = tuple[1]
    A.append(vals)

print(A)
print(components)
x = np.linalg.solve(np.array(A), b)
