import sys
import numpy as np
import random
import math

# import ros_numpy
from cv_bridge import CvBridge
import rosbag
import json
import cv2
# import rosbag_pandas

##########################################
#                                        #
#                OPENING                 #
#                                        #
##########################################

def save_file(name, data):
    with open('{}.json'.format(name), 'wb') as f:
        json.dump(data, f)

def parse_ackermann(ackermann):
    d = ackermann.drive
    return d.steering_angle

# bag = rosbag.Bag('5-13-17.bag')
laser = {}
ackermann = {}
camera = {}

TOPICS = {
    '/sensors/scan': 'laser',
    '/sensors/usb_cam/image_raw': 'camera',
    '/vesc/ackermann_cmd_mux/input/teleop': 'drive',
}

# print 'reading bag'
camera_messages = 0
total_camera_messages = 712
# bridge = CvBridge()

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    blur_size = 3
    image = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
    # v = np.median(image)
    v = np.mean(image)
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
 
    # return the edged image
    return edged

# maybe instead, do nsecs and every training epoch, look up the appropriate data?

# last_image = None
# for topic, msg, t in bag.read_messages(topics=TOPICS.keys()):
#     seconds = t.secs
#     if TOPICS[topic] == 'laser':
#         if seconds in laser:
#             laser[seconds].append(msg.ranges + msg.intensities)
#         else:
#             laser[seconds] = [msg.ranges + msg.intensities]
#     if TOPICS[topic] == 'camera':
#         camera_messages += 1
#         imgdata = bridge.imgmsg_to_cv2(msg)
#         imgdata = cv2.resize(imgdata[:,:,0], (0, 0), fx=0.25, fy=0.25)
#         if seconds not in camera:
#             camera[seconds] = [imgdata.flatten().tolist()]
#             if last_image is not None:
#                 camera[seconds-1].append(last_image.flatten().tolist())
#                 last_image = None
#             print float(camera_messages)/float(total_camera_messages)
#         else:
#             last_image = imgdata
#     if TOPICS[topic] == 'drive':
#         if seconds in ackermann:
#             ackermann[seconds].append(parse_ackermann(msg))
#         else:
#             ackermann[seconds] = [parse_ackermann(msg)]

# bag.close()
# print 'bag closed'

def save_file(name, data):
    with open('{}.json'.format(name), 'wb') as f:
        json.dump(data, f)

def open_file(name):
    with open('{}.json'.format(name), 'rb') as f:
        return json.load(f)

# save_file('dict_data/laser', laser)
# save_file('dict_data/ackermann', ackermann)
# save_file('dict_data/camera', camera)

##########################################
#                                        #
#                PARSING                 #
#                                        #
##########################################

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

##########################################
#                                        #
#                TRAINING                #
#                                        #
##########################################

import tensorflow as tf

# Parameters
# learning_rate = 1
training_iters = 1000
batch_size = 5
# display_step = training_iters/10
display_step = 1

# Network Parameters
n_input = 57600 # 1280/4 * 720/4
n_output = 1
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=5):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 320, 180, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    print conv1.get_shape()
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    print conv1.get_shape()

    fc1 = tf.reshape(conv1, [-1, weights['fc'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['fc']), biases['fc'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
conv_size = 10
l1_size = 32
full_size = 128
weights = {
    # 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([conv_size, conv_size, 1, l1_size])),
    # fully connected, IDK inputs - how did it get 2178, 1024 outputs
    'fc': tf.Variable(tf.random_normal([32*18 * l1_size, full_size])),
    # full_size inputs, 1 output
    'out': tf.Variable(tf.random_normal([full_size, n_output]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([l1_size])),
    'fc': tf.Variable(tf.random_normal([full_size])),
    'out': tf.Variable(tf.random_normal([n_output]))
}

# Construct model
print 'constructing model...'
pred = conv_net(x, weights, biases, keep_prob)
print 'model constructed'

# Define loss and optimizer
cost = tf.reduce_mean(tf.squared_difference(pred, y))
learning_rate = tf.placeholder(tf.float32, shape=[])
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
print 'optimizer created'

# Evaluate model- how to do for regression?
# correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
print 'init created'

max_rate = 1
min_rate = 0.001
def get_learning_rate(last_loss):
    if last_loss is None:
        return 1
    rate = min(last_loss/1000.0, max_rate)
    return max(rate, min_rate)

# Launch the graph
last_loss = None
saver = tf.train.Saver()
save_step = 10
with tf.Session() as sess:
    print 'initializing'
    sess.run(init)
    print 'session launched'
    # Keep training until reach max iterations
    for step in range(training_iters):
        targets, features = batch(batch_size)
        # Run optimization op (backprop)
        # print 'batch created'
        sess.run(optimizer, feed_dict={x: features, y: targets,
                                       keep_prob: dropout,
                                       learning_rate: get_learning_rate(last_loss)})
        if step % display_step == 0:
            # Calculate batch loss (not accuracy yet)
            loss = sess.run(cost, feed_dict={x: features,
                                             y: targets,
                                             keep_prob: 1.})
            last_loss = loss
            print("Iter {}, Minibatch Loss={}".format(step, loss))
        if step % save_step == 0:
            saver.save(sess, 'models/camera.ckpt')
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    # print("Testing Accuracy:", \
    #     sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
    #                                   y: mnist.test.labels[:256],
    #                                   keep_prob: 1.}))