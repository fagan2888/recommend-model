import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

training_steps = 10000
display_step = 1000

# Network Parameters
num_input = 22 # how many features
timesteps = 90 # use last three month's data
num_hidden = 128 # hidden layer num of features
num_output = 5 # input feature for the first dense layer
keep_prob = 1

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
    df = pd.concat([sh300, zz500, sp500, hsi, gold], 1).fillna(method = 'pad')
    df = df.pct_change().dropna()
    #summary_data = pd.concat([sh300.close.pct_change(),zz500.close.pct_change(),sp500.close.pct_change(),\
    #                         hsi.close.pct_change(),gold.close.pct_change()], 1)
    #summary_data.to_csv('asset.csv', index_label = 'date')

    total_num = len(df)
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for i in range(total_num//2):
        train_x.append(df.iloc[i:i+time_steps, :].values)
        train_y.append(df[['close']].iloc[i+time_steps+20, :].values)

    for i in range(total_num//2, total_num-time_steps):
        test_x.append(df.iloc[i:i+time_steps, :].values)
        test_y.append(df[['close']].iloc[i+time_steps, :].values)

    train_index = df.index[time_steps-1: total_num//2+time_steps-1]
    test_index = df.index[total_num//2+time_steps-1: total_num-1]
    
    return dict(train = train_x, test = test_x), dict(train = train_y, test = test_y), \
            dict(train_index = train_index, test_index = test_index)

def RNN(x, weights, biases):

    x = tf.unstack(x, timesteps, 1)

    def _lstm_cell():
        lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob = keep_prob, output_keep_prob = keep_prob)
        return lstm_cell

    cell = rnn.MultiRNNCell([_lstm_cell() for _ in range(1)])

    outputs, states = rnn.static_rnn(cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + biases['out']

feature, ret, idx = generate_index_data('2005-01-04', '2017-11-09', 90)

with tf.name_scope('lstm_layer') as scope:
    lstm_layer = RNN(X, weights, biases)

#lstm_layer = lstm_layer*tf.expand_dims((tf.div(1.0,tf.reduce_sum(lstm_layer,1))),1)

global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(1e-4, global_step, decay_steps = 1000, decay_rate=0.5,staircase=False)
real_ret = tf.reduce_prod(tf.add(tf.reduce_sum(tf.multiply(Y, lstm_layer), 1), 1.0))
panelty_1 = 1000*tf.reduce_sum(tf.square(tf.add(-1.0,tf.reduce_sum(lstm_layer, 1))))
panelty_2 = 1000*tf.reduce_sum(tf.add(tf.abs(lstm_layer),-lstm_layer))
panelty = panelty_1+panelty_2
final_ret = -real_ret + panelty
#final_ret = 10*tf.reduce_sum(tf.square(tf.add(-1.0,tf.reduce_sum(lstm_layer, 1))))

#train_step = tf.train.AdamOptimizer(learning_rate).minimize(final_ret, global_step = global_step)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(final_ret, global_step = global_step)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    
    for step in range(1, training_steps + 1):
        sess.run(train_step, feed_dict = {X: feature['train'], Y: ret['train']})
        tmp_ret_train = sess.run(real_ret, feed_dict = {X: feature['train'], Y: ret['train']})
        tmp_panelty_train = sess.run(panelty, feed_dict = {X: feature['train'], Y: ret['train']})
        tmp_res = sess.run(final_ret, feed_dict = {X: feature['train'], Y: ret['train']})
        tmp_lr = sess.run(learning_rate, feed_dict = {X: feature['train'], Y: ret['train']})
        tmp_weight_train = sess.run(lstm_layer, feed_dict = {X: feature['train'], Y: ret['train']})
        tmp_ret_test = sess.run(real_ret, feed_dict = {X: feature['test'], Y: ret['test']})
        tmp_weight_test = sess.run(lstm_layer, feed_dict = {X: feature['test'], Y: ret['test']})
        tmp_panelty_test = sess.run(panelty, feed_dict = {X: feature['test'], Y: ret['test']})
        #print sess.run(tmp_weight, feed_dict = {X: feature['train'], Y: ret['train']}).round(2)
        #print tmp_weight_test
        print 'Step: %d, tmp_ret_train: %.4f, final_ret: %.4f, tmp_panelty_train: %.4f, learning_rate: %f'%\
                (step, tmp_ret_train, tmp_res, tmp_panelty_train, tmp_lr)
        print 'Step: %d, tmp_ret_test: %.4f, tmp_panelty: %.4f'%\
                (step, tmp_ret_test, tmp_panelty_test)

        if step % display_step == 0:
            df_train = pd.DataFrame(tmp_weight_train, index = idx['train_index'], columns = ['w_sh300', 'w_zz500', 'w_sp500', 'w_hsi', 'w_gold'])
            df_test = pd.DataFrame(tmp_weight_test, index = idx['test_index'], columns = ['w_sh300', 'w_zz500', 'w_sp500', 'w_hsi', 'w_gold'])
            df_train.to_csv('./result_data/tf_alloc_train.csv', index_label = 'date')
            df_test.to_csv('./result_data/tf_alloc_test.csv', index_label = 'date')

if __name__ == '__main__':
    generate_index_data('2005-01-04', '2017-11-07', 90)


