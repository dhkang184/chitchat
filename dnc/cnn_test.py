# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 02:03:05 2017

@author: dhyun
"""

import tensorflow as tf
import numpy as np

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

x_train = np.load('/home/dhyun/class/2017.02/mnist2_2dimg.npy')
y_train = np.load('/home/dhyun/class/2017.02/mnist2_onehot.npy')
x_test = np.load('/home/dhyun/class/2017.02/test_mnist2_img.npy')
y_test = np.load('/home/dhyun/class/2017.02/test_mnist2_onehot.npy')

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 45])
keep_prob = tf.placeholder(tf.float32)


W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))

L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)

L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob)



W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 256], stddev=0.01))
L3 = tf.reshape(L2, [-1, 7 * 7 * 64])
L3 = tf.matmul(L3, W3)
L3 = tf.nn.relu(L3)
L3 = tf.nn.dropout(L3, keep_prob)


W4 = tf.Variable(tf.random_normal([256, 45], stddev=0.01))
model = tf.matmul(L3, W4)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(len(x_train) / batch_size)

for epoch in range(15):
    total_cost = 0

    for i in range(total_batch):
        #batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs= x_train[i*batch_size: (i+1)*batch_size]
        batch_ys = y_train[i*batch_size: (i+1)*batch_size]         
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)

        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X: batch_xs,
                                          Y: batch_ys,
                                          keep_prob: 0.7})
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print("succes")

#########
#
######
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('acc:', sess.run(accuracy,
                        feed_dict={X: x_test.reshape(-1, 28, 28, 1),
                                   Y: y_test,
                                   keep_prob: 1}))