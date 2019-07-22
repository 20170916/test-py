#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: lo time:2019-07-21
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, activation_function=None):
    # 定义weight变量矩阵，初始为随机变量会比全部为0好
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # 定义biases变量列表，推荐初始值不为0
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

    wx_b = tf.matmul(inputs, weights) + biases
    if activation_function is None:
        outputs = wx_b
    else:
        outputs = activation_function(wx_b)
    return outputs


def build_neural_layer():
    # 构建一个x_data,-1到1，有300行，每行一个特性
    # [:, np.newaxis]将行向量转成列向量
    x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
    # y=x的平方-0.5，并加入噪声
    # 噪声在0到0.05之间，跟x_data一样的格式
    noise = np.random.normal(0, 0.05, x_data.shape)
    y_data = np.square(x_data) - 0.5 + noise

    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])

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
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                        reduction_indices=[1]))

    # 使用优化器进行训练,来减少误差
    # 学习效率通常小于1，这里指定为0.1;即每一个练习步骤，都通过优化器，以0.1的学习效率对误差进行提升
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 初始化所有变量
    if int((tf.__version__).split('.')[1]) < 12:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    session = tf.Session()
    session.run(init)

    # 生成一个图片匡
    fig = plt.figure()
    # 连续画图,编号1，1，1
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data)
    # plt.show 会暂停整个程序，ion可以连续的plt
    # 老版本是在show方法指定Block=false
    plt.ion()
    plt.show()

    for i in range(1000):
        session.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            print(session.run(loss, feed_dict={xs: x_data, ys: y_data}))

            try:
                # catch掉第一次抹除时没有线
                ax.lines.remove(lines[0])
                pass
            except Exception:
                pass

            # 每50步显示一次预测数据
            prediction_value = session.run(prediction, feed_dict={xs: x_data})
            # 将prediction的值以曲线的形式plot上去
            # 红色的线：r-
            # 宽度为5：lw=5

            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
            # 抹除上次画的线
            # ax.lines.remove(lines[0])
            # 暂停0.1秒后继续
            plt.pause(0.1)

    return None


if __name__ == "__main__":
    build_neural_layer()
