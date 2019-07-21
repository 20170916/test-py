#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: lo time:2019-07-21
import tensorflow as tf


def place_holder():
    # 定义两个place_holder
    # 需要指定一个type
    # 也可以指定placeholder的结构，例如tf.placeholder(tf.float32,[2,2])指定2行2列的结构
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)

    # 乘法运算
    output = tf.multiply(input1, input2)

    with tf.Session() as session:
        # 以feed_dice形式传入两个placeholder的值
        print(session.run(output, feed_dict={input1: [7.], input2: [3.]}))

    return None


if __name__ == "__main__":
    place_holder()
