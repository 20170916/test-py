#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: lo time:2019-07-14

import tensorflow as tf
import numpy as np


def test():

    # create data,y=0.1x+0.3
    # 生成100个随机数列，指定为float32数据类型，tf中大部分数据的type都是float32
    x_data = np.random.rand(100).astype(np.float32)
    # 需要预测的值
    y_data = x_data*0.1 + 0.3

    # create tensorflow structure start
    # 定义一个参数变量，用随机数列生成-1到1取值的一维参数来初始化这个变量
    Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    # Weights = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
    # 定义初始值是0
    biases = tf.Variable(tf.zeros([1]))

    y = Weights*x_data + biases

    loss = tf.reduce_mean(tf.square(y-y_data))
    # 建立一个优化器，并用优化器减少误差；
    # 每一步训练都要做的事情，减少误差，提升参数的准确度，使下一次误差更小 ；这是神经网络的key idea
    # 可以选择多种优化器，这里使用梯度下降的优化器
    # 学习效率一般是一个小于1的数
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    # 初始化神经网络中的变量
    init = tf.initialize_all_variables()
    # create tensorflow structure end ###

    # session指神经网络中的一次会话，用run指向这次session中想要处理的地方，并激活这个地方
    sess = tf.Session()
    # 激活整个神经网络
    sess.run(init)

    for step in range(201):
        # 开始训练
        sess.run(train)
        # 每隔20步打印结果
        if step % 20 == 0:
            print(step, sess.run(Weights), sess.run(biases))


if __name__ == "__main__":
    test()
