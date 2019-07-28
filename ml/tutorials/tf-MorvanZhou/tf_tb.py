#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: lo time:2019-07-28

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            # 定义weight变量矩阵，初始为随机变量会比全部为0好
            weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
        with tf.name_scope('biases'):
            # 定义biases变量列表，推荐初始值不为0
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        with tf.name_scope('wx_plus_b'):
            wx_plus_b = tf.matmul(inputs, weights) + biases
        if activation_function is None:
            outputs = wx_plus_b
        else:
            outputs = activation_function(wx_plus_b)
        return outputs


def build_neural_layer():
    # define placeholder for inputs to network
    with tf.name_scope('inputs'):
        xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
        ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

    # 定义第一个隐藏层,
    # in_size是x_data的size就是1
    # out_size,给隐藏层定义10个神经元，输出10个size
    # 激励函数使用tf的relu
    l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

    # 定义输出层
    # 第一层的输出值作为输出层的输入值
    # in_size就是隐藏层的size，即10
    # out_size就是y_data的size，即1
    prediction = add_layer(l1, 10, 1, activation_function=None)

    # 计算损失,误差的平方求和后的平均值
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                        reduction_indices=[1]))

    with tf.name_scope('train'):
        # 使用优化器进行训练,来减少误差
        # 学习效率通常小于1，这里指定为0.1;即每一个练习步骤，都通过优化器，以0.1的学习效率对误差进行提升
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 初始化所有变量
    if int((tf.__version__).split('.')[1]) < 12:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    session = tf.Session()

    # 把整个框架加载到一个文件中,即graph的信息放到logs目录下
    # writer = tf.train.SummaryWriter("logs/", session.graph)
    writer = tf.summary.FileWriter("logs/", session.graph)
    session.run(init)

    return None


if __name__ == "__main__":
    build_neural_layer()
