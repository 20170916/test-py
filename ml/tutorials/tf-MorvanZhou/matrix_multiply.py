#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: lo time:2019-07-21
import tensorflow as tf


def matrix_multiply():
    matrix1 = tf.constant([[3, 3]])
    matrix2 = tf.constant([[2],
                           [2]])
    # matrix multiply
    product = tf.matmul(matrix1, matrix2)

    # session每run一次，tf才会执行一次定义的结构，这是rf的思考模式
    # with 语句运行完后会帮我们关闭session
    with tf.Session() as session:
        result = session.run(product)
        print(result)
    return None


if __name__ == "__main__":
    matrix_multiply()
