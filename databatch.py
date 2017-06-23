import sys
import numpy as np
import random
import math
import json

def open_file(f):
    with open('{}.json'.format(f), 'rb') as out:
        return json.load(out)

def flatten(data):
    ret = []
    for time in sorted(data):
        ret += data[time]
    return ret

print 'opening files'
data = {}
data['laser'] = flatten(open_file('normalized_data/laser'))
data['ackermann'] = flatten(open_file('normalized_data/ackermann'))
data['ackermann_cdf'] = flatten(open_file('normalized_data/ackermann_cdf'))
print 'files opened'

def next_x_values(arr, initial_idx, x=10):
    ret = [arr[initial_idx]]
    i = 1
    while len(arr)-1 > initial_idx and i < x:
        initial_idx += 1
        i += 1
        ret.append(arr[initial_idx])
    return ret

max_batch_size = len(data['laser'])
laser_length = 271
laser_datapoints = float(max_batch_size)
def batch(batch_size, next_x=10, cdf=True):
    ackermann_string = 'ackermann'
    if cdf:
        ackermann_string = 'ackermann_cdf'
    ackermann_datapoints = len(data[ackermann_string])
    batch_size = min(max_batch_size, batch_size)
    targets = []
    features = []
    samples = random.sample(range(max_batch_size), batch_size)
    for idx in samples:
        laser_single = data['laser'][idx][:laser_length]

        rel_idx = float(idx)/laser_datapoints

        ackermann_idx = int(math.ceil(rel_idx * ackermann_datapoints))

        next_values = next_x_values(data[ackermann_string], ackermann_idx, x=next_x)
        ackermann_mean = np.average(next_values, weights=np.arange(1, len(next_values)+1)[::-1])

        targets.append([ackermann_mean])

        features.append(laser_single)
    return (features, targets)