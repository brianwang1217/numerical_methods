import numpy as np
from operator import itemgetter

''' Fix the code '''
def consecutive_elem_ratio(seq):
    temp = [0]*len(seq)
    for i in range(len(seq)):
        temp[i] = (float(seq[i])/seq[i-1])
    seq.clear()
    seq += temp
    seq.pop(0)

fibonnaci_seq = [1, 1, 2, 3, 5, 8, 13, 21]
consecutive_elem_ratio(fibonnaci_seq)
print(fibonnaci_seq)

''' Sparse Matrix-vector multiplication '''
Ax = np.zeros(shape[0])

for (key, value) in A.items():
    for (k, v) in value.items():
    	Ax[key] += v * x[k]

''' Counting hashtags '''
hashtags = []
for tweet in tweets:
    tweet = tweet.lower().split()
    for word in tweet:
        if word[0] == '#':
            hashtags.append(word)

dict = {}
for hashtag in hashtags:
    if hashtag in dict:
        dict[hashtag] += 1
    else:
        dict[hashtag] = 1

hashtag_count_list = list(dict.items())
hashtag_counts = sorted(hashtag_count_list, key=lambda element:(-element[1], element[0]))
print(hashtag_counts)
