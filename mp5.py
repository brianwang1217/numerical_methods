import pandas as pd
import numpy as np
import np.linalg as la

''' Page 3 '''
tumor_data = pd.io.parsers.read_csv("breast-cancer-train.dat", header=None, names=labels)
copy = tumor_data.copy()
A_linear = np.zeros((300, 30))
del copy["patient ID"]
del copy["Malignant/Benign"]
#print(copy.iloc[0])
print(copy)
for i in range(300):
    a = 0
    while (a < 30):
        A_linear[i][a] = copy.values[i][a]
        a += 1

''' Page 4 '''
tumor_data = pd.io.parsers.read_csv("breast-cancer-train.dat", header=None, names=labels)
copy = tumor_data.copy()
A_quad = np.zeros((300, 14))
#print(copy.iloc[0])
#print(copy)

#print(subset_labels)
for i in range(300):
    A_quad[i][0] = tumor_data[subset_labels[0]][i]
    A_quad[i][1] = tumor_data[subset_labels[1]][i]
    A_quad[i][2] = tumor_data[subset_labels[2]][i]
    A_quad[i][3] = tumor_data[subset_labels[3]][i]
    A_quad[i][4] = tumor_data[subset_labels[0]][i] ** 2
    A_quad[i][5] = tumor_data[subset_labels[1]][i] ** 2
    A_quad[i][6] = tumor_data[subset_labels[2]][i] ** 2
    A_quad[i][7] = tumor_data[subset_labels[3]][i] ** 2
    A_quad[i][8] = tumor_data[subset_labels[0]][i] * tumor_data[subset_labels[1]][i]
    A_quad[i][9] = tumor_data[subset_labels[0]][i] * tumor_data[subset_labels[2]][i]
    A_quad[i][10] = tumor_data[subset_labels[0]][i] * tumor_data[subset_labels[3]][i]
    A_quad[i][11] = tumor_data[subset_labels[1]][i] * tumor_data[subset_labels[2]][i]
    A_quad[i][12] = tumor_data[subset_labels[1]][i] * tumor_data[subset_labels[3]][i]
    A_quad[i][13] = tumor_data[subset_labels[2]][i] * tumor_data[subset_labels[3]][i]

''' Page 5 '''
tumor_data = pd.io.parsers.read_csv("breast-cancer-train.dat", header=None, names=labels)
tumor = tumor_data["Malignant/Benign"]
print(tumor)
b = []
for x in tumor:
    if x == 'M':
        b.append(1.0)
    else:
        b.append(-1.0)
b = (np.array(b))

''' Page 6 '''
# Read in the data files
tumor_data = pd.io.parsers.read_csv("breast-cancer-train.dat", header=None, names=labels)
tumor_valid = pd.io.parsers.read_csv("breast-cancer-validate.dat", header=None, names=labels)

# Construct your A matrices
copy = tumor_data.copy()
del copy["patient ID"]
del copy["Malignant/Benign"]

valid_copy = tumor_valid.copy()
del valid_copy["patient ID"]
del valid_copy["Malignant/Benign"]

A_linear = np.zeros(copy.shape)
A_quad = np.zeros((300, 14))
valid_linear = np.zeros(valid_copy.shape)
valid_quad = np.zeros((len(valid_copy.index), 14))

for i in range(300):
    a = 0
    while (a < 30):
        A_linear[i][a] = copy.values[i][a]
        a += 1
for i in range(len(tumor_data.index)):
    A_quad[i][0] = tumor_data[subset_labels[0]][i]
    A_quad[i][1] = tumor_data[subset_labels[1]][i]
    A_quad[i][2] = tumor_data[subset_labels[2]][i]
    A_quad[i][3] = tumor_data[subset_labels[3]][i]
    A_quad[i][4] = tumor_data[subset_labels[0]][i] ** 2
    A_quad[i][5] = tumor_data[subset_labels[1]][i] ** 2
    A_quad[i][6] = tumor_data[subset_labels[2]][i] ** 2
    A_quad[i][7] = tumor_data[subset_labels[3]][i] ** 2
    A_quad[i][8] = tumor_data[subset_labels[0]][i] * tumor_data[subset_labels[1]][i]
    A_quad[i][9] = tumor_data[subset_labels[0]][i] * tumor_data[subset_labels[2]][i]
    A_quad[i][10] = tumor_data[subset_labels[0]][i] * tumor_data[subset_labels[3]][i]
    A_quad[i][11] = tumor_data[subset_labels[1]][i] * tumor_data[subset_labels[2]][i]
    A_quad[i][12] = tumor_data[subset_labels[1]][i] * tumor_data[subset_labels[3]][i]
    A_quad[i][13] = tumor_data[subset_labels[2]][i] * tumor_data[subset_labels[3]][i]

for i in range(len(tumor_valid.index)):
    a = 2
    while a < 32:
        valid_linear[i][a-2] = tumor_valid.values[i][a]
        a += 1

for i in range(len(tumor_valid.index)):
    valid_quad[i][0] = tumor_valid[subset_labels[0]][i]
    valid_quad[i][1] = tumor_valid[subset_labels[1]][i]
    valid_quad[i][2] = tumor_valid[subset_labels[2]][i]
    valid_quad[i][3] = tumor_valid[subset_labels[3]][i]
    valid_quad[i][4] = tumor_valid[subset_labels[0]][i] ** 2
    valid_quad[i][5] = tumor_valid[subset_labels[1]][i] ** 2
    valid_quad[i][6] = tumor_valid[subset_labels[2]][i] ** 2
    valid_quad[i][7] = tumor_valid[subset_labels[3]][i] ** 2
    valid_quad[i][8] = tumor_valid[subset_labels[0]][i] * tumor_valid[subset_labels[1]][i]
    valid_quad[i][9] = tumor_valid[subset_labels[0]][i] * tumor_valid[subset_labels[2]][i]
    valid_quad[i][10] = tumor_valid[subset_labels[0]][i] * tumor_valid[subset_labels[3]][i]
    valid_quad[i][11] = tumor_valid[subset_labels[1]][i] * tumor_valid[subset_labels[2]][i]
    valid_quad[i][12] = tumor_valid[subset_labels[1]][i] * tumor_valid[subset_labels[3]][i]
    valid_quad[i][13] = tumor_valid[subset_labels[2]][i] * tumor_valid[subset_labels[3]][i]
# Construct your b's
tumor = tumor_data["Malignant/Benign"]
tumor_valid = tumor_valid["Malignant/Benign"]
b = []
b_valid = []

for val in tumor:
    if val == 'M':
        b.append(1.0)
    else:
        b.append(-1.0)
for val in tumor_valid:
    if val == 'M':
        b_valid.append(1.0)
    else:
        b_valid.append(-1.0)
b = np.array(b)
b_valid = np.array(b_valid)
# Solve the least squares problem
lU, lE, lV = np.linalg.svd(A_linear, full_matrices=False)
qU, qE, qV = np.linalg.svd(A_quad, full_matrices=False)

coeff_linear = np.dot(lU.T, b)
weight_linear = np.divide(coeff_linear, lE)
weights_linear = np.dot(lV.T, weight_linear)

coeff_quad = np.dot(qU.T, b)
weight_quad = np.divide(coeff_quad, qE)
weights_quad = np.dot(qV.T, weight_quad)

# See how well your model (i.e. weights) does on the validate data set
p_linear = valid_linear @ weights_linear
p_quad = valid_quad @ weights_quad
# Plot a bar graph of the false-positives and false-negatives
fp_linear = 0
fn_linear = 0
fp_quad = 0
fn_quad = 0

for i in range(len(p_linear)):
    if p_linear[i] > 0 and b_valid[i] == -1:
        fp_linear += 1
    elif p_linear[i] < 0 and b_valid[i] == 1:
        fn_linear += 1

for i in range(len(p_quad)):
    if p_quad[i] > 0 and b_valid[i] == -1:
        fp_quad += 1
    elif p_quad[i] < 0 and b_valid[i] == 1:
        fn_quad += 1

bar_graph(fp_linear, fn_linear, fp_quad, fn_quad)
