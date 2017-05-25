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
data['ackermann'] = flatten(open_file('normalized_data/ackermann_v2'))
print 'files opened'

def next_x_values(arr, initial_idx, x=30):
    # if initial_idx == len(arr):
    #     return [arr[initial_idx-1]]
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
ackermann_datapoints = len(data['ackermann'])
def batch(batch_size):
    batch_size = min(max_batch_size, batch_size)
    targets = []
    features = []
    samples = random.sample(range(max_batch_size), batch_size)
    for idx in samples:
        laser_single = data['laser'][idx][:laser_length]

        rel_idx = float(idx)/laser_datapoints

        # should this be floor?
        ackermann_idx = int(math.ceil(rel_idx * ackermann_datapoints))

        next_values = next_x_values(data['ackermann'], ackermann_idx)
        ackermann_mean = np.average(next_values, weights=np.arange(1, len(next_values)+1)[::-1])

        targets.append([ackermann_mean])

        features.append(laser_single)
    return (features, targets)

# always pick previous or next?  probably next, but actually...

# make changes one at a time
# average steering values instead of laser values - and average future steering values