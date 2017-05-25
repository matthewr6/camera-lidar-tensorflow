import sys
import numpy as np
import random
import math
import json

laser_only = False
camera_only = True
verbose_batch = False
very_verbose_batch = False

def open_file(f):
    with open('{}.json'.format(f), 'rb') as out:
        return json.load(out)

print 'opening files'
data = {}
data['laser'] = open_file('normalized_data/laser')
data['ackermann_v2'] = open_file('normalized_data/ackermann_v2')
data['ackermann'] = open_file('normalized_data/ackermann')
data['camera'] = open_file('normalized_data/camera')
print 'files opened'

# let's use ackermann data as the base and fill in from there by weighted averaging of camera and laser info

def get_relatived_value(arr, rel_idx):
    forward_bias, idx = math.modf(rel_idx * len(arr))
    if idx+1 == len(arr):
        return arr[-1]
    first_bias = 1 - forward_bias
    v1 = np.array(arr[int(idx)]) * first_bias
    v2 = np.array(arr[int(idx)+1]) * forward_bias
    return (v1 + v2).tolist()
    # return np.array(arr[int(idx)+1])

def shuffle_data(features, targets):
    zipped = zip(features, targets)
    random.shuffle(zipped)
    return zip(*zipped)

# max_batch_size = len(data['ackermann'].keys())
max_batch_size = len(data['laser'].keys())
laser_length = 271
def batch(batch_size, mode='both', laser_cdf=True):
    ackermann_string = 'ackermann_v2'
    if not laser_cdf:
        ackermann_string = 'ackermann'
    batch_size = min(max_batch_size, batch_size)
    targets = []
    features = []
    times = random.sample(data['laser'].keys(), batch_size)
    for idx, time in enumerate(times):
        item_count = len(data['laser'][time])
        if time not in data[ackermann_string]:# or time not in data['camera']:
            continue
        for second_idx, value in enumerate(data['laser'][time]):
            value = value[:laser_length]
            rel_idx = float(second_idx)/item_count
            ackermann_single = get_relatived_value(data[ackermann_string][time], rel_idx)
            targets.append([ackermann_single])
            features.append(value)
        if verbose_batch and very_verbose_batch:
            print 'batch prep percent', float(idx)/batch_size
    if verbose_batch:
        'returning batch size={}'.format(batch_size)
    if mode == 'both':
        features = zip(*features)
    return (features, targets)

# always pick previous or next?  probably next, but actually...

# make changes one at a time
# average steering values instead of laser values - and average future steering values