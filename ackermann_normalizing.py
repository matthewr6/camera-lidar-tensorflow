import numpy as np
import json
import matplotlib.pyplot as plt 
import sys
from scipy.special import erfc

with open('dict_data/ackermann.json', 'rb') as f:
    original = json.load(f)

data = []
for time in original:
    data += original[time]

x = np.sort(data)
y = np.linspace(0, 1, len(data), endpoint=False)

def get_value(v):
    w = np.where(x==v)
    s = np.sum([y[i] for i in w])
    return float(s)/len(w[0])

y_vals = []
for t in x:
    v = get_value(t)
    if v not in y_vals:
        y_vals.append(v)

x = np.unique(x)
y = y_vals
mapping = {}
for idx, v in enumerate(x):
    mapping[v] = y[idx]

# def get_closest(arr, v):
#     return (np.abs(np.array(arr)-v).argmin())
import math
def get_relatived_value(arr, rel_idx):
    forward_bias, idx = math.modf(rel_idx * len(arr))
    if idx+1 == len(arr):
        return arr[-1]
    first_bias = 1 - forward_bias
    v1 = np.array(arr[int(idx)]) * first_bias
    v2 = np.array(arr[int(idx)+1]) * forward_bias
    return (v1 + v2).tolist()

epsilon = 1e-9
def denormalize(value):
    if value >= 1:
        value = 1-epsilon
    return get_relatived_value(x, value)

if __name__ == '__main__':
    for time in original:
      original[time] = [100.0*mapping[v] for v in original[time]]
    with open('normalized_data/ackermann_v2.json', 'wb') as f:
        json.dump(original, f)

# # thoughts
# take this cdf (x is value, y is probability at)
# for each value of x (in data) return the probability (Y) possibly multiplied by some scalar