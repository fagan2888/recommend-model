import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python import debug as tf_debug

training_steps = 10000
display_step = 1
ih = 30 #invest horizon

# Network Parameters
num_input = 27 # how many features
timesteps = 90 # use last three month's data
num_hidden = 128 # hidden layer num of features
num_output = 5 # input feature for the first dense layer
keep_prob = 1
learning_rate = 10.0

weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_output]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_output]))
}

X = tf.placeholder('float', [None, timesteps, num_input])
Y = tf.placeholder('float', [None, num_output])

def generate_index_data(start, end, time_steps):
    sh300 = pd.read_csv('data/sh300.csv', index_col = 0, parse_dates = True).loc[start:end].dropna(1)
    zz500 = pd.read_csv('data/zz500.csv', index_col = 0, parse_dates = True).loc[start:end].dropna(1)
    sp500 = pd.read_csv('data/sp500.csv', index_col = 0, parse_dates = True).loc[start:end].dropna(1)
    hsi = pd.read_csv('data/hsi.csv', index_col = 0, parse_dates = True).loc[start:end].dropna(1)
    gold = pd.read_csv('data/gold.csv', index_col = 0, parse_dates = True).loc[start:end].dropna(1)
    sh300['pct_change'] = sh300['close'].pct_change()
    zz500['pct_change'] = zz500['close'].pct_change()
    sp500['pct_change'] = sp500['close'].pct_change()
    hsi['pct_change'] = hsi['close'].pct_change()
    gold['pct_change'] = gold['close'].pct_change()
    sh300['std'] = sh300['pct_change'].rolling(ih).std()
    zz500['std'] = zz500['pct_change'].rolling(ih).std()
    sp500['std'] = sp500['pct_change'].rolling(ih).std()
    hsi['std'] = hsi['pct_change'].rolling(ih).std()
    gold['std'] = gold['pct_change'].rolling(ih).std()
    df = pd.concat([sh300, zz500, sp500, hsi, gold], 1).fillna(method = 'pad').fillna(0.0)
    df['sh300_pct_change_30'] = sh300['close'].pct_change(ih).shift(-ih).fillna(0.0)
    df['zz500_pct_change_30'] = zz500['close'].pct_change(ih).shift(-ih).fillna(0.0)
    df['sp500_pct_change_30'] = sp500['close'].pct_change(ih).shift(-ih).fillna(0.0)
    df['hsi_pct_change_30'] = hsi['close'].pct_change(ih).shift(-ih).fillna(0.0)
    df['gold_pct_change_30'] = gold['close'].pct_change(ih).shift(-ih).fillna(0.0)
    del df['close']

    total_num = len(df)
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for i in range(total_num//2):
        train_x.append(df.iloc[i:i+time_steps, :-5].values)
        train_y.append(df.iloc[i+time_steps-1, -5:].values)

    for i in range(total_num//2, total_num-time_steps+1):
        test_x.append(df.iloc[i:i+time_steps, :-5].values)
        test_y.append(df.iloc[i+time_steps-1, -5:].values)

    zero_arr_train = np.zeros_like(train_y)
    zero_arr_train[np.arange(len(train_y)), np.argmax(train_y, 1)] = 1
    train_y = zero_arr_train
    zero_arr_train = np.zeros_like(train_y)

    zero_arr_test = np.zeros_like(test_y)
    zero_arr_test[np.arange(len(test_y)), np.argmax(test_y, 1)] = 1
    test_y = zero_arr_test

    return dict(train=train_x, test=test_x), dict(train=train_y, test=test_y)

def RNN(x, weights, biases):

    x = tf.unstack(x, timesteps, 1)

    def _lstm_cell():
        lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob = keep_prob, output_keep_prob = keep_prob)
        return lstm_cell

    cell = rnn.MultiRNNCell([_lstm_cell() for _ in range(1)])

    outputs, states = rnn.static_rnn(cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + biases['out']

feature, best = generate_index_data('2005-01-04', '2017-11-09', 90)

with tf.name_scope('lstm_layer') as scope:
    logits = RNN(X, weights, biases)
    logits = tf.identity(logits, name = "logits")

with tf.name_scope('output_layer') as scope:
    prediction = tf.nn.softmax(logits, name = 'prediction')

    global_step = tf.Variable(0)
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = Y))
    lr = tf.train.exponential_decay(learning_rate, global_step, decay_steps = 500, decay_rate=0.5,staircase=False)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = lr)
    train_step = optimizer.minimize(loss_op, global_step = global_step)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1), name = 'correct_pred')
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name = 'accuracy')
with tf.Session() as sess:
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    init = tf.global_variables_initializer()
    sess.run(init)

    for step in range(1, training_steps + 1):
        sess.run(train_step, feed_dict = {X: feature['train'], Y: best['train']})
        if step % display_step == 0 or step == 1:
            with tf.name_scope('result') as scope:
                loss, acc, pred, tmp_lr = sess.run([loss_op, accuracy, prediction, lr], feed_dict = {X: feature['train'], Y: best['train']})
                loss_test, acc_test, pred_test, tmp_lr_test = sess.run([loss_op, accuracy, prediction, lr], feed_dict = {X: feature['test'], Y: best['test']})
            print("Train Step " + str(step) + ", Loss= " + \
                  "{:.4f}".format(loss) + ", Accuracy= " + \
                  "{:.3f}".format(acc), "learning rate=", "{:f}".format(tmp_lr))
            print("Test Step " + str(step) + ", Loss= " + \
                  "{:.4f}".format(loss_test) + ", Accuracy= " + \
                  "{:.3f}".format(acc_test), "learning rate=", "{:f}".format(tmp_lr_test))
