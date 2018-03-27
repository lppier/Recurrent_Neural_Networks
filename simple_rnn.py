# Credits go to : https://github.com/ageron/handson-ml

import tensorflow as tf
import numpy as np
import os


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


n_inputs = 3
n_neurons = 5

reset_graph()

X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, [X0, X1],
                                                dtype=tf.float32)
Y0, Y1 = output_seqs

init = tf.global_variables_initializer()

X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]])
X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]])

with tf.Session() as sess:
    init.run()
    Y0, Y1 = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X1_batch})

print("Output of Y0 : ")
print(Y0)

print("Output of Y1 : ")
print(Y1)

# Use dynamic un-rolling through time to achieve the same thing

reset_graph()

n_steps = 2  # the rest is the same as before
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
init = tf.global_variables_initializer()

X_batch = np.array([
    # t0        #t1
    [[0, 1, 2], [9, 8, 7]],  # instance 1
    [[3, 4, 5], [0, 0, 0]],  # instance 2
    [[6, 7, 8], [6, 5, 4]],  # instance 3
    [[9, 0, 1], [3, 2, 1]],  # instance 4
])

with tf.Session() as sess:
    init.run()
    outputs_val = outputs.eval(feed_dict={X: X_batch})

print(outputs_val)

# Suppose we want instance 2 to have only 1 input sequence (step 0)
# This is useful like in sentences, no all sequences will have the same length.
seq_length_batch = np.array([2, 1, 2, 2])

reset_graph()

n_steps = 2  # the rest is the same as before
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
seq_len = tf.placeholder(tf.float32, [None])
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32, sequence_length=seq_len)
init = tf.global_variables_initializer()

X_batch = np.array([
    # t0        #t1
    [[0, 1, 2], [9, 8, 7]],  # instance 1
    [[3, 4, 5], [0, 0, 0]],  # instance 2 <-- only has 1 input, instead of 2
    [[6, 7, 8], [6, 5, 4]],  # instance 3
    [[9, 0, 1], [3, 2, 1]],  # instance 4
])

with tf.Session() as sess:
    init.run()
    outputs_val = outputs.eval(feed_dict={X: X_batch, seq_len: seq_length_batch})

print(outputs_val)
