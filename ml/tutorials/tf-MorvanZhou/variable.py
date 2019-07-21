#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: lo time:2019-07-21

import tensorflow as tf


def variable():
    # 定义一个变量，初始值和名字
    state = tf.Variable(0, name='counter')
    print(state.name)

    # 定义一个常量，变量+常量结果还是变量
    one = tf.constant(1)
    new_value = tf.add(state, one)

    # 变量赋值
    update = tf.assign(state, new_value)

    # 初始化所有变量
    init = tf.initialize_all_variables()

    with tf.Session() as session:
        # 激活所有变量
        session.run(init)
        # 进行3次上面定义的操作
        for _ in range(3):
            # 启动update功能
            session.run(update)
            # 通过session获取state变量的值
            print(session.run(state))


    return None


if __name__ == "__main__":
    variable()
