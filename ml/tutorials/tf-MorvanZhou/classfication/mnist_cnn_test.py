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


def compute_accuracy(xs, v_xs, ys, v_ys, prediction, sess, keep_prob):
    # 计算准确度
    # global prediction
    # 将xs feed到prediction中，生成预测值,1行10列，0到1的概率值
    # y_pre = session.run(prediction, feed_dict={xs: v_xs})
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})

    # 比对最大概率与真实值是否匹配
    # correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 得到最终输出的百分比
    # result = session.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
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
    # stride [1, x_movement, y_movement, 1]
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

    # add output layer

    prediction = add_layer(xs, 784, 10,  activation_function=tf.nn.softmax)

    # the error between prediction and real data
    # 分类问题一般使用cross_entropy算法，交叉熵损失函数
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                  reduction_indices=[1]))
    # 使用梯度下降优化器，学习率0.5
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

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
        # 提取一部分x,y作为样本；随机梯度下降
        # 计算能力有限，每次不学习整套data，学习整套data会非常耗时
        # 这里每次学习100个数据
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
        # 每隔50步打印准确度
        #
        if i % 50 == 0:
            # 使用测试集计算准确度
            print(compute_accuracy(
                xs, mnist.test.images, ys, mnist.test.labels, prediction, sess))


if __name__ == "__main__":
    mnist_cnn_test()
