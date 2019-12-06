import numpy as np
import matplotlib.pyplot as plt
import math

''' Page 1 '''
put = lambda S: np.maximum(0, 40 - S)
plt.title("put option as specific strike price")
plt.plot(np.arange(80), [put(x) for x in range(80)])

''' Page 2 '''
S0 = 1.0
r = 0.06
sigma = 0.25
t = 1.0

def St(S0, r, sigma, t):
    St = S0 * math.e ** ((r - (sigma ** 2) / 2) * t + np.random.normal(0, math.sqrt(t) * sigma))
    return St

''' Page 3 '''
def calc_payout(S, sigma, r, T, K):
    #Implement this
    St = generate_brownian_asset_price(S, sigma, r, T)
    payout = call_payout(St, K)
    return payout

''' Page 4 '''
expected_arr = np.zeros(samples)
for i in range(samples):
    St = S * math.e ** ((r - (sigma ** 2) / 2) * T + np.random.normal(0, math.sqrt(T) * sigma))
    expected_arr[i] = max(0, St - K)

P_T = np.average(expected_arr)
call = max(0, T - 40)
discount_factor = math.e ** (-r * T)
price_now  = P_T * discount_factor
