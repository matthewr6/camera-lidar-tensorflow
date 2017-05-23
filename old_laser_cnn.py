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

def shuffle_data(features, targets):
    zipped = zip(features, targets)
    random.shuffle(zipped)
    return zip(*zipped)

max_batch_size = len(data['ackermann'].keys())
laser_length = 271
def batch(batch_size, mode='both'):
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
            laser_single = get_relatived_value(data['laser'][time], rel_idx)[:laser_length]
            targets.append([value])
            if mode == 'laser':
                features.append(laser_single)
            elif mode == 'camera':
                features.append(get_relatived_value(data['camera'][time], rel_idx))
            else:
                camera_single = get_relatived_value(data['camera'][time], rel_idx)
                features.append([camera_single, laser_single])
        if verbose_batch and very_verbose_batch:
            print 'batch prep percent', float(idx)/batch_size
    if verbose_batch:
        'returning batch size={}'.format(batch_size)
    if mode == 'both':
        features = zip(*features)
    return (features, targets)

import tensorflow as tf

training_iters = 500
batch_size = 15
display_step = 1

n_input = 271
n_output = 1
dropout = 0.75


x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv1d(x, W, b, stride=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv1d(x, W, stride=stride, padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

# will I need maxpooling?
# def maxpool2d(x, k=2):
#     # MaxPool2D wrapper
#     return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
#                           padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    out = tf.reshape(x, shape=[-1, 271, 1])

    # Convolution Layer
    out = conv1d(out, weights['wc1'], biases['bc1'])
    # print conv1.get_shape()
    # Max Pooling (down-sampling)
    # conv1 = maxpool2d(conv1, k=2)
    # conv1 = conv1d(conv1, weights['wc2'], biases['bc2'])

    out = tf.reshape(out, [-1, weights['fc1'].get_shape().as_list()[0]])
    out = tf.add(tf.matmul(out, weights['fc1']), biases['fc1'])
    out = tf.nn.relu(out)
    # Apply Dropout?
    # fc1 = tf.nn.dropout(fc1, dropout)

    # out = tf.add(tf.matmul(out, weights['fc3']), biases['fc3'])
    # out = tf.nn.relu(out)

    # Output, class prediction
    out = tf.add(tf.matmul(out, weights['out']), biases['out'])
    return out

# Store layers weight & bias
# add more layers
# two middle layers
conv1_size = 4
l1_size = 12
l2_size = l1_size
full_size = 256
weights = {

    'wc1': tf.Variable(tf.random_normal([conv1_size, 1, l1_size])),

    'fc1': tf.Variable(tf.random_normal([271 * l2_size, full_size])),

    'out': tf.Variable(tf.random_normal([full_size, n_output]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([l1_size])),
    'fc1': tf.Variable(tf.random_normal([full_size])),
    'out': tf.Variable(tf.random_normal([n_output]))
}


print 'constructing model...'
pred = conv_net(x, weights, biases, keep_prob)
print 'model constructed'

cost = tf.reduce_mean(tf.squared_difference(pred, y))
learning_rate = tf.placeholder(tf.float32, shape=[])
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
print 'optimizer created'

init = tf.global_variables_initializer()
print 'init created'

max_rate = 1.0
min_rate = 0.0
ackermann_scale = 100.0
def get_learning_rate(last_loss, past_losses):
    if last_loss is None:
        return 1.0
    rate = min(last_loss/100.0, max_rate)
    # if len(past_losses) == loss_history and np.std(past_losses) <= 0.1:
    #     rate = rate * 10.0
    return max(rate, min_rate)

last_loss = None
saver = tf.train.Saver()
save_step = 10
past_losses = []
loss_history = 5
lowest_loss = 1000.0
lowest_iter = 0

loss_history = []
with tf.Session() as sess:
    print 'initializing'
    sess.run(init)
    print 'session launched'
    for step in range(training_iters):
        features, targets = batch(batch_size, mode='laser')
        sess.run(optimizer, feed_dict={x: features, y: targets,
                                       keep_prob: dropout,
                                       learning_rate: get_learning_rate(last_loss, past_losses)})
        if step % display_step == 0:
            loss = sess.run(cost, feed_dict={x: features,
                                             y: targets,
                                             keep_prob: 1.})
            last_loss = loss
            lowest_loss = min(lowest_loss, loss)
            if lowest_loss == loss:
                lowest_iter = step
            # past_losses.append(loss)
            # if len(past_losses) > loss_history:
            #     past_losses = past_losses[1:]
            print("Iter {}, Minibatch avg error={}".format(step, np.sqrt(loss/(ackermann_scale**2.0))))
            # print("Iter {}, Minibatch avg error={}".format(step, loss))
            loss_history.append(np.sqrt(loss/(ackermann_scale**2.0)))
            # loss_history.append(loss)
        if step % save_step == 0:
            saver.save(sess, 'models/laser_cnn.ckpt')

# print lowest_iter, lowest_loss/(ackermann_scale**2.0)

if sys.argv[1]:
    with open('losses_{}.json'.format(sys.argv[1]), 'wb') as f:
        json.dump(loss_history, f)