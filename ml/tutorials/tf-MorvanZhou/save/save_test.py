#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: lo time:2019-08-02
import tensorflow as tf
import numpy as np


def save():
    # save to file
    # 保存神经网络的weight和biases
    # 变量需要定义dtype
    weight = tf.Variable([[1, 2, 3], [3, 4, 5]], dtype=tf.float32, name="weight")
    biases = tf.Variable([[1, 2, 3]], dtype=tf.float32, name="biases")

    # init variable
    init = tf.initialize_all_variables()

    # 定义saver来存储tf神经网络的变量
    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(init)
        # 保存
        save_path = saver.save(session, "my_net/save_net.ckpt")
        print("save to path:", save_path)

    return None


def restore():
    # restore variable
    # 必须定义相同shape和type的变量才能导入
    weight = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weight")
    biases = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")

    # 不必定义initialize，restore时会被initialize
    saver = tf.train.Saver()

    with tf.Session() as session:
        # 加载持久化的变量后，会对相同名字的变量进行复制
        saver.restore(session, "my_net/save_net.ckpt")
        print("weight:", session.run(weight))
        print("biases:", session.run(biases))
    return None


if __name__ == "__main__":
    # save()
    restore()
