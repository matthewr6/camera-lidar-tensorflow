import numpy as np
import random
import math
import json

def open_file(name):
    with open('{}.json'.format(name), 'rb') as f:
        return json.load(f)

print 'opening files'
data = {}
data['laser'] = open_file('dict_data/laser')
data['ackermann'] = open_file('dict_data/ackermann')
data['camera'] = open_file('dict_data/camera')
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

laser_only = False
camera_only = True
verbose_batch = False
very_verbose_batch = False

def shuffle_data(features, targets):
    zipped = zip(features, targets)
    random.shuffle(zipped)
    return zip(*zipped)

max_batch_size = len(data['ackermann'].keys())
def batch(batch_size):
    batch_size = min(max_batch_size, batch_size)
    targets = []
    features = []
    times = random.sample(data['ackermann'].keys(), batch_size)
    for idx, time in enumerate(times):
        item_count = len(data['ackermann'][time])
        if time not in data['laser'] or time not in data['camera']:
            continue
        for second_idx, value in enumerate(data['ackermann'][time]):
            rel_idx = float(second_idx)/item_count
            laser_single = get_relatived_value(data['laser'][time], rel_idx)
            targets.append([value])
            if laser_only:
                features.append(laser_single)
            elif camera_only:
                features.append(get_relatived_value(data['camera'][time], rel_idx))
            else:
                camera_single = get_relatived_value(data['camera'][time], rel_idx)
                features.append(camera_single + laser_single)
        if verbose_batch and very_verbose_batch:
            print 'batch prep percent', float(idx)/batch_size
    if verbose_batch:
        'returning batch size={}'.format(batch_size)
    return (targets, features)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.core import Reshape
import keras.backend as K

batch_size = 5
num_classes = 1

model = Sequential()
model.add(Reshape((320, 180, 1), input_shape=(320*180,)))
model.add(Conv2D(32, kernel_size=(10, 10), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.75))
model.add(Dense(num_classes))

def custom_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))

model.compile(loss=custom_loss,
              optimizer=keras.optimizers.Adam())
print 'compiled'

# https://github.com/fchollet/keras/issues/4446
# https://github.com/fchollet/keras/issues/68

# https://keras.io/getting-started/faq/#how-can-i-use-keras-with-datasets-that-dont-fit-in-memory

# https://github.com/fchollet/keras/issues/3223


targets, features = batch(1)
print 'batch done'
loss = model.train_on_batch(features, targets)
print loss