#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: lo time:2019-02-17
import unittest
from datetime import date

from sklearn.datasets import load_iris, fetch_20newsgroups, load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

        # 特征工程（标准化）

        return None

    def test_variance(self):
        FBLocationTest.knncls()
