#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: lo time:2019-07-28
from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
# 没有数据时会先下载
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def add_layer(inputs, in_size, out_size, activation_function=None):
    # 定义weight变量矩阵，初始为随机变量会比全部为0好
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # 定义biases变量列表，推荐初始值不为0
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

    wx_plus_b = tf.matmul(inputs, weights) + biases
    if activation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activation_function(wx_plus_b)
    return outputs


def compute_accuracy(xs, v_xs, ys, v_ys, sess, keep_prob, prediction):
    # global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    # 定义卷积神经网络层
    # x时输入值，W是定义的weight变量
    # 定义个2维的卷积神经网络，输入为x，即图片的所有信息；w是权重变量；
    # strides是跨长，tf中是一个长度为4的列表，第一个和最后一个元素都要等于1，中间两个值对应水平跨度和垂直跨度
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    # padding有两种取值，same和valid；same抽取的图片与原图片大小相同，valid会小一点
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # 定义pooling图层
    # 用pooling处理跨度大的问题，保存更多的图片信息
    # stride [1, x_movement, y_movement, 1]
    # 输入参数x是卷积神经网络层输出的数据
    # Must have strides[0] = strides[3] = 1
    # stride在x，y位置多移动一位，隔2个像素移动一下，从而将整个图片的长宽压缩
    # 与conv2d不同在于不用传入weight
    # ksize是向下取样的维度
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def mnist_cnn_test():
    # define placeholder for inputs to network
    # 28x28=784
    # softmax一般用来做分类问题的，搭配交叉熵损失函数
    xs = tf.placeholder(tf.float32, [None, 784])
    ys = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    # 处理传入的图片信息
    # xs包括了所有图片的例子，-1是先忽略维度；28，28对应784拆分的像素点；1对应黑白图片，彩色的rgb是3
    x_image = tf.reshape(xs, [-1, 28, 28, 1])
    # x_image 的shape：[n_samples, 28, 28, 1]
    # print(x_image.shape)

    # 定义各层
    # conv1 layer
    # 定义第一个卷积神经网络的weight，patch:5*5,in size:1,out size 32
    # patch 是扫描器大小，in size是image的厚度，out size是输出的高
    W_conv1 = weight_variable([5, 5, 1, 32])
    # 32个长度的bias
    b_conv1 = bias_variable([32])
    # 搭建第一层卷积神经网络，并嵌套一个激励函数relu的非线性的处理，使其非线性化
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 28x28x32
    h_pool1 = max_pool_2x2(h_conv1)  # output size 14x14x32

    # conv2 layer
    W_conv2 = weight_variable([5, 5, 32, 64])  # patch 5x5, in size 32, out size 64
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14x14x64
    h_pool2 = max_pool_2x2(h_conv2)  # output size 7x7x64

    # fc1 layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    # 将上面两层卷积神经网络加工出来的3维数据转成1维
    # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    ## fc2 layer ##
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # the error between prediction and real data
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                  reduction_indices=[1]))  # loss
    # 不再使用梯度下降优化器，adam需要更小的学习率
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    sess = tf.Session()
    # important step
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
        if i % 50 == 0:
            print(compute_accuracy(
                xs, mnist.test.images[:1000], ys, mnist.test.labels[:1000], sess, keep_prob, prediction))


if __name__ == "__main__":
    mnist_cnn_test()
