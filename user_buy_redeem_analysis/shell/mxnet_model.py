#coding=utf8


from mxnet import autograd, nd
import mxnet as mx
from mxnet.gluon import loss as gloss
import random
import zipfile



if __name__ == '__main__':

    ctx = mx.gpu()
    print('will use', ctx)


    num_inputs = 2
    num_hidden = 2
    num_outputs = 3


    W_xh = nd.random.normal(scale = 0.01, shape = (num_inputs , num_hidden), ctx = ctx)
    W_hh = nd.random.normal(scale = 0.01, shape = (num_hidden , num_hidden), ctx = ctx)

    b_h = nd.zeros(num_hidden, ctx = ctx)

    W_hy = nd.random.normal(scale = 0.01, shape = (num_hidden, num_outputs), ctx = ctx)

    b_y = nd.zeros(num_outputs, ctx = ctx)

    params = [W_xh, W_hh, b_h, W_hy, b_y]

    for param in params:
        param.attach_grad()


    H = state

    W_xh, W_hh, b_h, W_hy, b_y = params

    outputs = []

    for X in inputs:
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(X, W_hy) + b_y

        outputs.append(Y)



