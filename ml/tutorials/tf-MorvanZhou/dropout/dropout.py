#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: lo time:2019-07-29
from __future__ import print_function
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# load data，使用sklearn的数据集
digits = load_digits()
# 加载x，从0-9数字的图片数据
X = digits.data
# 数组位置代表预测值
y = digits.target
y = LabelBinarizer().fit_transform(y)
# 将x，y分成训练数据和测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)


def add_layer(inputs, in_size, out_size, layer_name, activation_function=None, keep_prob_value=None,):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    # here to dropout
    # drop掉Wx_plus_b,输出更新后的Wx_plus_b
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob_value)

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
    # 记录output
    tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs


def dropout():
    # define placeholder for inputs to network
    # keep_prob，一直保持多少结果不被drop掉
    keep_prob = tf.placeholder(tf.float32)
    xs = tf.placeholder(tf.float32, [None, 64])  # 8x8
    ys = tf.placeholder(tf.float32, [None, 10])

    # add output layer
    # 添加两层，隐藏层和输出层
    # 隐藏层，sklearn中digits的x是64个单位，输出是10个单位，分别对应0-9
    # 激励函数使用tanh可避免信息变成null的问题
    l1 = add_layer(xs, 64, 50, 'l1', activation_function=tf.nn.tanh, keep_prob_value=keep_prob)
    prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax, keep_prob_value=keep_prob)

    # the loss between prediction and real data
    # 输出交叉熵
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                  reduction_indices=[1]))  # loss
    # 将cross_entropy当作loss记录下来
    tf.summary.scalar('loss', cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.Session()
    merged = tf.summary.merge_all()
    # summary writer goes in here
    # 记录两个summary，一个是训练的summary，一个是测试的summary
    # 使用tb显示loss的变化曲线
    train_writer = tf.summary.FileWriter("logs/train", sess.graph)
    test_writer = tf.summary.FileWriter("logs/test", sess.graph)

    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(500):
        # here to determine the keeping probability
        # keep_prob:0.5，保持50不被drop掉
        # sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.5})
        sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 1}) # 全部保留时，会出现过拟合问题
        if i % 50 == 0:
            # record loss
            # keep_drop:1,记录result时，不drop掉任何数据
            train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
            test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
            train_writer.add_summary(train_result, i)
            test_writer.add_summary(test_result, i)


if __name__ == "__main__":
    dropout()
