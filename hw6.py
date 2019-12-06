import numpy as np

''' CSR Matrix '''
A = np.zeros(A_csr.shape)
data_index = 0
for i in range(1, len(A_csr.indptr)):
    for x in range(A_csr.indptr[i - 1], A_csr.indptr[i]):
        A[i - 1][A_csr.indices[data_index]] = A_csr.data[data_index]
        data_index += 1
print(A)

''' Predict Popularity '''
ratings = np.zeros(k)
#single person ratings
for j in range(n):
    # each movie
    for i in range(len(ratings)):
        ratings[i] += np.sum(get_friend_prefs(j) * get_movie_attr(i)) / float(m)
max = 0
index = 0
for x in range(len(ratings)):
    if (ratings[x] > max):
        index = x
        max = ratings[x]
top = index

''' Model Popularity '''
prefs = []
A = []

# do A for all movies
for i in range(k):
    A.append(get_movie_attr(i))
A = np.array(A)

# calculate for each friend
for i in range(n):
    # do b for a friend
    b = get_ratings(i)

    # solve x for a friend
    x = np.linalg.solve(A, b)
    prefs.append(x)
prefs = np.transpose(prefs)
