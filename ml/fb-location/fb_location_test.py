#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: lo time:2019-02-17
import unittest
from datetime import date

from sklearn.datasets import load_iris, fetch_20newsgroups, load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd


class FBLocationTest(unittest.TestCase):

    @staticmethod
    def knncls():
        """
        K近邻预测用户签到位置
        :return:None
        """
        # 读取数据
        data = pd.read_csv("./data/train.csv")
        print(data.head(10))
        # 处理数据
        # 1 缩小数据，对查询数据进行筛选
        data = data.query("x > 1.0 & x < 1.25 &y > 2.5 & y < 2.75")

        # 处理时间数据
        time_value = pd.to_datetime(data['time'], unit='s')

        print(time_value)

        # 把日期数据格式转换成字典格式
        time_value = pd.DatetimeIndex(time_value)

        # 构造时间特征
        data["day"] = time_value.day
        data["hour"] = time_value.hour
        data["weekday"] = time_value.weekday

        # 把时间戳特征删除,pandas中 axis=1表示删除列，sklearn中axis=0表示删除列
        data = data.drop(["time"], axis=1)
        print(data)

        # 删除签到数量少于n的数据
        place_count = data.groupby("place_id").count()
        tf = place_count[place_count.row_id > 3].reset_index()
        data = data[data["place_id"].isin(tf.place_id)]

        print(data)

        # 取出数据中的特征值和目标
        y = data["place_id"]
        x = data.drop(["place_id"], axis=1)

        # 分割训练集和测试集
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

        # 特征工程（标准化）
        std = StandardScaler()

        # 算法流程
        knn = KNeighborsClassifier(n_neighbors=5)
        # fit predict score
        knn.fit(x_train, y_train)
        # 得出预测结果
        y_predict = knn.predict(x_test)
        print("预测的目标签到位置为：", y_predict)
        # 输出预测准确率
        print("预测准确率：", knn.score(x_test, y_test))

        return None

    def test_variance(self):
        FBLocationTest.knncls()
