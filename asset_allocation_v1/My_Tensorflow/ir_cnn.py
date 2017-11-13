from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import io
from PIL import Image
import shutil
import numpy as np

import tensorflow as tf
from data_processing import generate_ir_data
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
plt.style.use('seaborn')

filepath = './data/indic.csv'
feature, ir = generate_ir_data(filepath)
loop_num = 10000

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

def plot(ir, res, step, dtype):
    ir = ir.ravel()
    res = res.ravel()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    #ax2 = ax1.twinx()
    ax1.plot(range(len(ir)), ir, color = 'b', label = 'ori_data')
    #ax2.plot(range(len(res)), res, color = 'r', label  = 'fit_data')
    ax1.plot(range(len(res)), res, color = 'r', label  = 'fit_data')
    ax1.legend(loc = 2)
    #ax2.legend(loc = 1)
    ax1.legend(loc = 1)
    fig.savefig('tmp/tmp_%s_%d.png'%(dtype, step))
    with open('tmp/tmp_%s_%d.png'%(dtype, step)) as f:
        picture = f.read()  

    image = tf.image.decode_png(picture, channels = 4)
    image = tf.expand_dims(image, 0)
    '''
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    picture = Image.open(buf)
    buf.close()
    image = tf.image.decode_png(picture, channels = 4)
    image = tf.expand_dims(image, 0)
    '''
    return image

x = tf.placeholder(tf.float32, [None, 12, 29])
x_image = tf.reshape(x, [-1, 12, 29, 1])
with tf.name_scope('conv_layer1') as scope:
    W_conv1 = weight_variable([5,5,1,16])
    b_conv1 = bias_variable([16])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

with tf.name_scope('dense_layer1') as scope:
    W_fc1 = weight_variable([12*29*16, 1024])
    b_fc1 = bias_variable([1024])
    h_pool1_flat = tf.reshape(h_conv1, [-1, 12*29*16])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('dense_layer2') as scope:
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    dense = tf.nn.sigmoid(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

with tf.name_scope('dense_layer3') as scope:
    W_fc3 = weight_variable([10, 1])
    b_fc3 = bias_variable([1])
    y_conv = tf.matmul(dense, W_fc3) + b_fc3

global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(1e-2, global_step, decay_steps = 20, decay_rate=0.99,staircase=True)
y_ = tf.placeholder(tf.float32, [None, 1])
#mse = tf.reduce_mean(tf.square(y_ - y_conv))
mse = tf.losses.mean_squared_error(y_, y_conv)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(mse,global_step = global_step)
#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(mse,global_step = global_step)

tf.summary.scalar('ir_mse', mse)
tf.summary.scalar('ir_bias', tf.reduce_sum(b_fc3))
merged_summary = tf.summary.merge_all()

os.system('pkill tensorboard')
os.system('nohup tensorboard --logdir=/home/ipython/yaojiahui/My_Tensorflow/logs>/home/ipython/yaojiahui/My_Tensorflow.nohup.out 2>&1 &')

train_image = tf.placeholder(tf.uint8, [1, 550, 800, 4])
train_image_op = tf.summary.image('train_result', train_image)

test_image = tf.placeholder(tf.uint8, [1, 550, 800, 4])
test_image_op = tf.summary.image('test_result', test_image)

if os.path.isdir('./logs/ir/ir_train'):
    shutil.rmtree('./logs/ir/ir_train')
    os.mkdir('./logs/ir/ir_train')
if os.path.isdir('./logs/ir/ir_test'):
    shutil.rmtree('./logs/ir/ir_test')
    os.mkdir('./logs/ir/ir_test')
if os.path.isdir('./tmp'):
    shutil.rmtree('./tmp')
    os.mkdir('./tmp')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_writer = tf.summary.FileWriter('./logs/ir/ir_train', sess.graph)
test_writer = tf.summary.FileWriter('./logs/ir/ir_test')
for i in range(loop_num):
    sess.run(train_step, feed_dict={x:feature['train'], y_:ir['train'], keep_prob: 0.8})
    train_summary = sess.run(merged_summary, feed_dict = {x:feature['train'], y_:ir['train'], keep_prob: 1})
    train_writer.add_summary(train_summary, i)

    test_summary = sess.run(merged_summary, feed_dict = {x:feature['test'], y_:ir['test'], keep_prob: 1})
    test_writer.add_summary(test_summary, i)

    if i in (range(0, loop_num, loop_num//10) + [loop_num-1]):
        print('step: %d, train_mse: %f, test_mse: %f'%(i, sess.run(mse, feed_dict={x:feature['train'], y_:ir['train'], keep_prob: 1}),\
                sess.run(mse, feed_dict={x:feature['test'], y_:ir['test'], keep_prob: 1})))
        y_arr_train = sess.run(y_, feed_dict = {x:feature['train'], y_:ir['train'], keep_prob: 1})
        y_conv_arr_train = sess.run(y_conv, feed_dict = {x:feature['train'], y_:ir['train'], keep_prob: 1})
        train_image_run = plot(y_arr_train, y_conv_arr_train, i, 'train')
        train_image_summary = sess.run(train_image_op, feed_dict = {train_image:sess.run(train_image_run)})
        train_writer.add_summary(train_image_summary, i)

        y_arr_test = sess.run(y_, feed_dict = {x:feature['test'], y_:ir['test'], keep_prob: 1})
        y_conv_arr_test = sess.run(y_conv, feed_dict = {x:feature['test'], y_:ir['test'], keep_prob: 1})
        test_image_run = plot(y_arr_test, y_conv_arr_test, i, 'test')
        test_image_summary = sess.run(test_image_op, feed_dict = {test_image:sess.run(test_image_run)})
        test_writer.add_summary(test_image_summary, i)

sess.close()

