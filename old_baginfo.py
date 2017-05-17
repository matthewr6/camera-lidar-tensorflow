import sys
import numpy as np
import random

# import ros_numpy
from cv_bridge import CvBridge
import rosbag
import json
import cv2
# import rosbag_pandas

def save_file(name, data):
    with open('{}.json'.format(name), 'wb') as f:
        json.dump(data, f)

# http://answers.ros.org/question/191003/can-rosbag-filter-create-a-new-bag-file-with-multiple-topics-within-it/
# ['/scan', '/usb_cam/image_raw', '/vesc/ackermann_cmd_mux/output']

def parse_ackermann(ackermann):
    d = ackermann.drive
    # steering angle velocity
    # learn steering angle rate, and put limit on steering angle on +- 10 (maybe too little)
    # use current steering angle to predict steering angle velocity - RNN?  sorta recurrent
    # disregard speed for now.  do a separate predictor.

    # possible - longer term angle for "turn corner"?
    # predict "3 feet ahead" point constantly and feed that to a lower system to move
    # RNN to predict current steering angle
    # this is far future.

    # for now, use current steering angle and sensor data to predict steering angle velocity
    # red layer has most information on camera
    # camera data - every X pixels?  skip pixels/lines
    # infinite/nan - how to represent that.
    # RVR regression
    # return (d.steering_angle, d.steering_angle_velocity, d.speed, d.acceleration, d.jerk)
    # return d.steering_angle_velocity
    # return d.steering_angle.
    return d.steering_angle

class Data:
    def __init__(self, data, times):
        self.data = data
        self.times = times

# bag = rosbag.Bag('5-6-2017bag-nosteering-filtered.bag')
bag = rosbag.Bag('5-13-17.bag')
laser_data = []
laser_times = []
ackermann_data = []
ackermann_times = []
camera_data = []
camera_times = []

# http://docs.ros.org/jade/api/ackermann_msgs/html/msg/AckermannDrive.html
TOPICS = {
    '/sensors/scan': 'laser',
    '/sensors/usb_cam/image_raw': 'camera',
    # '/vesc/ackermann_cmd_mux/output': 'drive',
    '/vesc/ackermann_cmd_mux/input/teleop': 'drive',
}

print 'reading bag'
camera_messages = 0
total_camera_messages = 712
import time
bridge = CvBridge()

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

for topic, msg, t in bag.read_messages(topics=TOPICS.keys()):
    if TOPICS[topic] == 'laser':
        #to align w/camera data.
        laser_data.append(msg.ranges + msg.intensities) # do we want intensities? - yes
        laser_times.append(t.secs)
    if TOPICS[topic] == 'camera':
        camera_messages += 1
        if t.secs not in camera_times:
            b = time.clock()
            # print 'beginning'
            imgdata = bridge.imgmsg_to_cv2(msg)
            red_data = cv2.resize(imgdata[:,:,0], (0, 0), fx=0.25, fy=0.25)
            # cv2.imshow('image', auto_canny(red_data))
            # cv2.imshow('image', red_data)
            # cv2.waitKey()
            # imgdata = auto_canny(red_data)
            camera_data.append(imgdata.flatten().tolist())
            camera_times.append(t.secs)
            print 'percent through', float(camera_messages)/float(total_camera_messages), camera_messages
            # print 'time', time.clock() - b
            # print ''
    if TOPICS[topic] == 'drive':
        ackermann_data.append(parse_ackermann(msg))
        ackermann_times.append(t.secs)

bag.close()
print 'bag closed'
print len(camera_times)
print len(laser_times)

# data = {
#     'laser': Data(laser_data, laser_times),
#     # 'camera': Data(camera_data, camera_times),
#     'ackermann': Data(ackermann_data, ackermann_times),
# }
# # print data['ackermann'].data
# # print data['laser'].data

# # let's choose camera as the "base" because that's when we first get stuff
# # print 'filling/parsing'
# # start_time = camera_times[0]
# start_time = laser_times[0]
# def fill_missingtimes(data, name):
#     new_data = [data.data[0]]
#     new_times = [data.times[0]]
#     idx = 1
#     curtime = data.times[idx] # this assumes it follows idx=0 but that's safe for now.
#     while curtime < data.times[-1]:
#         if curtime not in data.times: # missing data
#             new_data.append(new_data[-1])
#         else:
#             new_data.append(data.data[idx])
#             idx += 1
#         new_times.append(curtime)
#         curtime = min(curtime+1, data.times[idx])
#     return (new_data, new_times)

# # fill in blank times
# # this is weird because there are sometimes some large gaps.  also, it seems like there's a gap of 0 seconds in some places...
# for d in data:
#     firstidx = next(i for i,v in enumerate(data[d].times) if v >= start_time)
#     data[d].times = data[d].times[firstidx:]
#     new = fill_missingtimes(data[d], d)
#     data[d].data = new[0]
#     data[d].times = new[1]

# # trim excess times - is this necessary?  or should we just append times?
# max_time = np.array([data[d].times[-1] for d in data]).min()
# for d in data:
#     lastidx = next(i for i, v in enumerate(data[d].times) if v >= max_time)
#     data[d].times = data[d].times[:lastidx]
#     data[d].data = data[d].data[:lastidx]

# print len(data['ackermann'].data)
# print len(data['laser'].data)
# print len(np.unique(data['ackermann'].times))
# print len(np.unique(data['laser'].times))
# print set(data['laser'].times) - set(data['ackermann'].times)
# print np.unique(data['laser'].times)
# print np.unique(data['ackermann'].times)

# # check times and make sure they're equal
# thing = None
# for d in data:
#     if thing:
#         if not data[d].times == thing:
#             import sys;print 'time arrays are different!';sys.exit()
#     thing = data[d].times

# # print 'saving'
# # for d in data:
# #     print 'saving {}'.format(d)
# #     save_file('parsed/{}'.format(d), {
# #         'data': data[d].data,
# #         'times': data[d].times
# #     })

# # unsure how the drive system works...

# # recall
# # data = {
# #   'teleop': Data(teleop_data, teleop_times),
# #   'laser': Data(laser_data, laser_times),
# #   'speed': Data(speed_data, speed_times),
# #   'position': Data(position_data, position_times),
# #   'ackermann': Data(ackermann_data, ackermann_times),
# # }
# # ackermann is one of:
# # (d.steering_angle, d.steering_angle_velocity, d.speed, d.acceleration, d.jerk)

# def combine_features(featuredict, include, count):
#     combined = []
#     for i in range(count):
#         app = np.array([])
#         for row in featuredict:
#             if row in include:
#                 rowdata = featuredict[row].data[i]
#                 if type(rowdata) in [type(()), type([])]:
#                     for d in rowdata:
#                         app = np.append(app, d)
#                 else:
#                     app = np.append(app, rowdata)
#         combined.append(app)
#     return np.array(combined)

# def flatten_tuples(arr):
#     combined = []
#     for row in arr:
#         newrow = []
#         for tup in row:
#             for item in tup:
#                 newrow.append(item)
#         combined.append(newrow)
#     return np.array(combined)

# # this produces an array of 1dimensional array of all laser data
# # array of 1dimensional arrays of all teleop data
# # targets = flatten_tuples(data['ackermann'].data)
# features = combine_features(data, ['laser'], len(data['ackermann'].times))
# targets = data['ackermann'].data
# print targets

# def shuffle_data(features, targets):
#     zipped = zip(features, targets)
#     random.shuffle(zipped)
#     return zip(*zipped)

# features, targets = shuffle_data(features, targets)
# test_set_size = 10
# train_X = np.array(features[:-test_set_size])
# train_y = np.array(targets[:-test_set_size]).reshape(-1, 1)
# test_X = np.array(features[-test_set_size:])
# test_y = np.array(targets[-test_set_size:]).reshape(-1, 1)

# use_nn = False
# train_model = True

# # #RVM stuff
# if train_model:
#     if not use_nn:
#         from skrvm import RVR

#         clf = RVR(kernel='rbf')
#         # print features
#         # print targets
#         clf.fit(train_X, train_y)
#         print clf.score(test_X, test_y)
#         for i in range(len(test_X)):
#             print clf.predict(np.array(test_X[i]).reshape(1, -1)), np.array(test_y[i]).reshape(1, -1)

#     #Tensorflow stuff
#     if use_nn:
#         import tensorflow as tf

#         # train_y = np.array([[y] for y in train_y])
#         # test_y = np.array([[y] for y in test_y])

#         x_size = train_X.shape[1]
#         print 'x_size', x_size
#         n_hidden_1 = x_size # 1st layer number of features
#         y_size = train_y.shape[1]

#         X = tf.placeholder("float", shape=[None, x_size])
#         y = tf.placeholder("float", shape=[None, y_size])

#         # Create model
#         def multilayer_perceptron(x, weights, biases):
#             # Hidden layer with RELU activation
#             layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
#             layer_1 = tf.nn.relu(layer_1)
#             # Hidden layer with RELU activation
#             # layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
#             # layer_2 = tf.nn.relu(layer_2)
#             # Output layer with linear activation
#             out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
#             return out_layer

#         weights = {
#             'h1': tf.Variable(tf.random_normal([x_size, n_hidden_1])),
#             # 'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
#             'out': tf.Variable(tf.random_normal([n_hidden_1, y_size]))
#         }
#         biases = {
#             'b1': tf.Variable(tf.random_normal([n_hidden_1])),
#             # 'b2': tf.Variable(tf.random_normal([n_hidden_2])),
#             'out': tf.Variable(tf.random_normal([y_size]))
#         }

#         predict = multilayer_perceptron(X, weights, biases)

#         # Define loss and optimizer
#         cost = tf.reduce_mean(tf.squared_difference(predict, y))

#         learning_rate = 0.01
#         updates = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#         # Run SGD
#         with tf.Session() as sess:
#             init = tf.global_variables_initializer()
#             sess.run(init)

#             iterations = 5000
#             display_info = iterations/10
#             for epoch in range(iterations):
#                 _, c = sess.run([updates, cost], feed_dict={X: train_X, y: train_y})


#                 if epoch % display_info == 0:
#                     train_accuracy = tf.reduce_mean(tf.squared_difference(predict, y)).eval({X: train_X, y: train_y})
#                     test_accuracy  = tf.reduce_mean(tf.squared_difference(predict, y)).eval({X: test_X, y: test_y})
#                     print "Epoch = {}, cost = {}, train accuracy = {}, test accuracy = {}".format(
#                         epoch + 1,
#                         c,
#                         100. * train_accuracy,
#                         100. * test_accuracy)
#                     # print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))

#             test_score = tf.reduce_mean(tf.squared_difference(predict, y))
#             print "Testing score mean squared:", test_score.eval({X: test_X, y: test_y})