import numpy as np

''' Page 2 '''
M = np.zeros((len(links), len(links)))
print(len(links))
count = 0
x = 0
y = 0
for link in links:
    for page in link:
        M[int(page)][int(count)] += (1.0 / len(link))
    count += 1

''' Page 3 '''
# Fill in this function
def get_next_state(links, current_state):
    # Do not change the below line (you must use np.zeros_like)
    next_state = np.zeros_like(current_state)
    # Fill in code to correctly get next state'
    for i in range(len(links)):
        prob = float(1.0 / len(links[i]))
        for j in range(len(links[i])):
            next_state[links[i][j]] = next_state[links[i][j]] + current_state[i] * prob
    return next_state

# Run code to get result after one iteration and save result in first_state
first_state = get_next_state(links, initial_state)

''' Page 4 '''
stable_state = initial_state.copy()
while np.linalg.norm((get_next_state(links, stable_state) - stable_state), np.inf) > 10 ** -9:
    stable_state = get_next_state(links, stable_state)

''' Page 5 '''
page_rankings = sorted(zip(titles, stable_state), key=lambda x: x[1])
page_rankings = page_rankings[-50:]
page_rankings = page_rankings[::-1]
print(page_rankings)
