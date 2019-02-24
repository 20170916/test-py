#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: lo time:2019-02-22
import unittest
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class BostonHousePricePredict(unittest.TestCase):

    @staticmethod
    def predict():
        """
        线性回归预测波士顿房价
        :return:
        """

        # 1 获取数据
        lb = load_boston( )

        # 2 分割数据集和训练集
        x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)
        print(y_train, y_test )

        # 3 进行标准化处理
        # (特征值和目标值都需要进行标准化处理)，实例化连个标准化api
        std_x = StandardScaler()
        x_train = std_x.fit_transform(x_train)
        x_test = std_x.transform(x_test)

        std_y = StandardScaler()
        # sk_learn 1.9版本需要二维数组，x_train.reshape（-1，1 ）
        y_train = std_y.fit_transform(y_train.reshape(-1, 1))
        y_test = std_y.transform(y_test.reshape(-1,1))

        # 4 estimator评估预测
        # 4.1 正规方程求解方式预测结果
        lr = LinearRegression()
        lr.fit(x_train, y_train)
        print(lr.coef_)

        # 预测测试集的房价
        y_lr_predict = std_y.inverse_transform(lr.predict(x_train))
        print("测试集里每个房子的预测价格 ", y_lr_predict )

        # 4.2 梯度下降进行房价预测
        sgd = SGDRegressor()
        sgd.fit(x_train, y_train)
        print(sgd.coef_)
        # 预测测试集的房价
        y_sgd_predict = std_y.inverse_transform(sgd.predict(x_test))
        print("sgd预测，测试集里每个房子的预测价格", y_sgd_predict )

        return None

    def test_decision(self):
        BostonHousePricePredict.predict()
