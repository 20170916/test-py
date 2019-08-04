#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: lo time:2019-08-04
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# set random seed for comparing the two result calculations
tf.set_random_seed(1)

# this is data
mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)

# hyperparameters
# 学习率
lr = 0.001
# 循环次数
training_iters = 100000
batch_size = 128

# 每行28个像素点，每次input就是这一行的28个像素
n_inputs = 28   # MNIST data input (img shape: 28*28)
# 28行的像素，每次输入一行的像素，28行就对应了28次
n_steps = 28    # time steps
n_hidden_units = 128   # neurons in hidden layer
# 图片分成10个类，0-9
n_classes = 10      # MNIST classes (0-9 digits)

# 定义x，y的placeHolder
# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
# rnn是一个cell，进入cell前，输入数据要经过一层隐藏层，隐藏层处理完再进入rnn的cell
# rnn的cell计算完结果后，再输出到output的隐藏层
# 相当与rnn的cell相比于普通nn多了两层隐藏层
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def RNN(X, weights, biases):
    # 先后定义rnn的输入隐藏层，cell和输出隐藏层，并返回result

    # hidden layer for input to cell
    ########################################

    # transpose the inputs shape from
    # X ==> (128 batch * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # into hidden
    # X_in = (128 batch * 28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell
    ##########################################

    # basic LSTM Cell.
    # forget_bias=1,初始时不希望forget过去的记忆
    # rnn处理时会保存每一步计算的结果，每一步的结果就是一个state
    # lstm的state分成主线的c_state和分线的h_state
    # 生成初始化state时，会生成一个元祖tuple，包含两个元素；state_is_tuple就是定义生成的是否时元祖
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    else:
        cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    # lstm cell is divided into two parts (c_state, h_state)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    # You have 2 options for following step.
    # 1: tf.nn.rnn(cell, inputs);
    # 2: tf.nn.dynamic_rnn(cell, inputs).
    # If use option 1, you have to modified the shape of X_in, go and check out this:
    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
    # In here, we go for option 2.
    # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
    # Make sure the time_major is changed accordingly.
    # 选择rnn时如何运算的，运算的结果时outputs和state，state是元祖
    # outputs是一个列表，每一步的output都存在这个outputs中
    # time_major的time对应step，每个step就是一个时间点；time_major指定x_in中的step是否在主要维度(主要维度就是第一维度)
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

    # hidden layer for output as the final results
    #############################################
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']

    # # or
    # unpack to list [(batch, outputs)..] * steps
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
    else:
        outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    # outputs[-1]是看完所有行后，最后一行的输出；以最后一行的输出来预测0-9
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']    # shape = (128, 10)

    return results


# rnn返回的results预测值
pred = RNN(x, weights, biases)
# 计算损失
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# 计算精确度
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        # 提取出下一个batch
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # 将图片流reshape成28行28列及batch_size的数列组
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
            }))
        step += 1

if __name__ == "__main__":
    print("ok")
