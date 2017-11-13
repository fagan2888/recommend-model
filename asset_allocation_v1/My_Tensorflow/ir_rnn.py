""" Recurrent Neural Network.

A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)

Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

import os
import shutil
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
#from data_processing import generate_ir_data
from tensorflow.contrib import rnn
from ipdb import set_trace

from data_processing import generate_ir_data_2
from utils import moving_average

plt.style.use('seaborn')

filepath = './data/indic7.csv'
#print(feature['test'])

# Training Parameters
#learning_rate = 0.001
training_steps = 10000
display_step = 200

# Network Parameters
num_input = 2 # how many features
timesteps = 90 # use last three month's data
num_hidden = 128 # hidden layer num of features
num_output = 16 # input feature for the first dense layer
start = 30*26

#summary&model dir
train_summary_dir = './logs/ir/ir_rnn_train_%s'%start
test_summary_dir = './logs/ir/ir_rnn_test_%s'%start
model_dir = './model/ir/ir_rnn_model_tanh_nci_%s'%start

#get ori_data
feature, ir, index = generate_ir_data_2(filepath, timesteps, start = start)

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, 1])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_output]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_output]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    def _lstm_cell():
        # Define a lstm cell with tensorflow
        lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

        # Add a Dropout wrapper to lstm_cell
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob = 0.9, output_keep_prob = 0.9)
        return lstm_cell
    
    #lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*2, state_is_tuple = True)
    #state = lstm_cell.zero_state(3, tf.float32)
    #outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, initial_state = state, dtype=tf.float32)

    # Get lstm cell output
    cell = rnn.MultiRNNCell([_lstm_cell() for _ in range(1)])
    outputs, states = rnn.static_rnn(cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def plot(ir, res, step, train_or_test):
    idx = index[train_or_test][29:]
    ir = ir.ravel()[29:]
    res = moving_average(res.ravel(), 30)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    #ax2 = ax1.twinx()
    #ax1.plot(range(len(ir)), ir, color = 'b', label = 'ori_data')
    ax1.plot(idx, ir, color = 'b', label = 'ori_data')
    #ax2.plot(range(len(res)), res, color = 'r', label  = 'fit_data')
    #ax1.plot(range(len(res)), res, color = 'r', label  = 'fit_data')
    ax1.plot(idx, res, color = 'r', label  = 'fit_data', alpha = 0.5)
    ax1.legend(loc = 2)
    #ax2.legend(loc = 1)
    ax1.legend(loc = 1)
    fig.savefig('tmp/tmp_%s_%d.png'%(train_or_test, step))
    f = open('tmp/tmp_%s_%d.png'%(train_or_test, step))
    picture = f.read()  
    f.close()

    image = tf.image.decode_png(picture, channels = 4)
    image = tf.expand_dims(image, 0)
    return image

with tf.name_scope('lstm_layer') as scope:
    lstm_layer = RNN(X, weights, biases)

'''
with tf.name_scope('dropout_layer') as scope:
    lstm_drop = tf.nn.dropout(lstm_layer, 0.7)
    '''
#first dense layer
with tf.name_scope('dense_layer1') as scope:
    W_fc1 = weight_variable([num_output, 1])
    b_fc1 = bias_variable([1])
    y_lstm = tf.matmul(lstm_layer, W_fc1) + b_fc1

'''
#Another dense layer
with tf.name_scope('dense_layer2') as scope:
    W_fc2 = weight_variable([10, 1])
    b_fc2 = bias_variable([1])
    y_lstm = tf.matmul(y_lstm, W_fc2) + b_fc2
    '''

global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(1e-2, global_step, decay_steps = 20, decay_rate=0.99,staircase=True)
mse = tf.losses.mean_squared_error(Y, y_lstm)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(mse, global_step = global_step)

with tf.name_scope('ir_rnn') as scope:
    tf.summary.scalar('irr_rnn_mse', mse)
    merged_summary = tf.summary.merge_all()

    train_image = tf.placeholder(tf.uint8, [1, 550, 800, 4])
    train_image_op = tf.summary.image('train_result', train_image)

    test_image = tf.placeholder(tf.uint8, [1, 550, 800, 4])
    test_image_op = tf.summary.image('test_result', test_image)

init = tf.global_variables_initializer()

os.system('pkill tensorboard')
os.system('nohup tensorboard --logdir=/home/ipython/yaojiahui/My_Tensorflow/logs>/home/ipython/yaojiahui/My_Tensorflow.nohup.out 2>&1 &')

if os.path.isdir(train_summary_dir):
    shutil.rmtree(train_summary_dir)
    os.mkdir(train_summary_dir)
if os.path.isdir(test_summary_dir):
    shutil.rmtree(test_summary_dir)
    os.mkdir(test_summary_dir)
if os.path.isdir('./tmp'):
    shutil.rmtree('./tmp')
    os.mkdir('./tmp')


with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
    test_writer = tf.summary.FileWriter(test_summary_dir)
    sess.run(init)

    for step in range(1, training_steps+1):
        sess.run(train_step, feed_dict={X: feature['train'], Y: ir['train']})

        if step % display_step == 0 or step == 1:
            train_mse = sess.run(mse, feed_dict={X: feature['train'], Y: ir['train']})
            print("Step " + str(step) + ", Training Mean Square Loss: " + "{:.4f}".format(train_mse))
            train_summary = sess.run(merged_summary, feed_dict={X: feature['train'], Y: ir['train']})
            train_writer.add_summary(train_summary, step)

            Y_train = sess.run(Y, feed_dict={X: feature['train'], Y: ir['train']})
            y_lstm_train = sess.run(y_lstm, feed_dict={X: feature['train'], Y: ir['train']})
            train_image_run = plot(Y_train, y_lstm_train, step, 'train') 
            train_image_summary = sess.run(train_image_op, feed_dict = {train_image: sess.run(train_image_run)})
            train_writer.add_summary(train_image_summary, step)

            test_mse = sess.run(mse, feed_dict={X: feature['test'], Y: ir['test']})
            print("Testing Mean Square Loss:", "{:.4f}".format(test_mse))
            test_summary = sess.run(merged_summary, feed_dict={X: feature['test'], Y: ir['test']})
            test_writer.add_summary(test_summary, step)

            Y_test = sess.run(Y, feed_dict={X: feature['test'], Y: ir['test']})
            y_lstm_test = sess.run(y_lstm, feed_dict={X: feature['test'], Y: ir['test']})
            test_image_run = plot(Y_test, y_lstm_test, step, 'test') 
            test_image_summary = sess.run(test_image_op, feed_dict = {test_image: sess.run(test_image_run)})
            test_writer.add_summary(test_image_summary, step)

            if train_mse < 0.05:
                break

    saver = tf.train.Saver()
    saver.save(sess, model_dir, global_step = step)
    with open('./result_data/train_result_1.npz', 'wb') as f:
        np.savez(
                f, 
                train_data = Y_train, \
                train_result = y_lstm_train,\
                test_data = Y_test, \
                test_result = y_lstm_test
                )

    print("Optimization Finished!")
