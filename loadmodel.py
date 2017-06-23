import numpy as np
import matplotlib.pyplot as plt

from databatch import batch
from ackermann_normalizing import denormalize

import tensorflow as tf


n_input = 271
n_output = 1
dropout = 1.0

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create some wrappers for simplicity
def conv1d(x, W, b, stride=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv1d(x, W, stride=stride, padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.elu(x)

# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    out = tf.reshape(x, shape=[-1, 271, 1])

    # Convolution Layer
    out = conv1d(out, weights['wc1'], biases['bc1'])
    out = conv1d(out, weights['wc2'], biases['bc2'])

    out = tf.reshape(out, [-1, weights['fc1'].get_shape().as_list()[0]])
    out = tf.add(tf.matmul(out, weights['fc1']), biases['fc1'])
    out = tf.nn.elu(out)

    out = tf.add(tf.matmul(out, weights['fc2']), biases['fc2'])
    out = tf.nn.elu(out)

    # Output, class prediction
    out = tf.add(tf.matmul(out, weights['out']), biases['out'])
    return out

conv1_size = 7
conv2_size = 7
l1_size = 24
l2_size = l1_size
full_size_1 = 64
full_size_2 = 32
full_size_3 = 32
weights = {

    'wc1': tf.Variable(tf.random_normal([conv1_size, 1, l1_size])),
    'wc2': tf.Variable(tf.random_normal([conv2_size, l1_size, l2_size])),

    'fc1': tf.Variable(tf.random_normal([271 * l2_size, full_size_1])),
    'fc2': tf.Variable(tf.random_normal([full_size_1, full_size_2])),
    'fc3': tf.Variable(tf.random_normal([full_size_2, full_size_3])),

    'out': tf.Variable(tf.random_normal([full_size_3, n_output]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([l1_size])),
    'bc2': tf.Variable(tf.random_normal([l2_size])),
    'fc1': tf.Variable(tf.random_normal([full_size_1])),
    'fc2': tf.Variable(tf.random_normal([full_size_2])),
    'fc3': tf.Variable(tf.random_normal([full_size_3])),
    'out': tf.Variable(tf.random_normal([n_output]))
}


print 'constructing model...'
pred = conv_net(x, weights, biases, keep_prob)
print 'model constructed'

cost = tf.reduce_mean(tf.squared_difference(pred, y))
learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
print 'optimizer created'

init = tf.global_variables_initializer()
print 'init created'
 
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, 'models/test.ckpt')

    features, targets = batch(1000000)
    predictions = sess.run(pred, feed_dict={x: features})
    x = [t[0]/(100) for t in targets] # remember the division by 100 for denormalization!
    y = [float(p[0])/100.0 for p in predictions]

    def bound(v):
        v = max(0.0, v)
        return min(1.0, v)

    fig, ax = plt.subplots()
    y = [bound(v) for v in y]
    x = [bound(v) for v in x]
    ax.scatter(x, y)

    lims = [
        0, 1
    ]
    fit = np.poly1d(np.polyfit(x, y, 1))
    ax.plot(lims, lims)
    ax.plot(x, fit(x))
    plt.show()