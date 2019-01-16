#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: lo time:2019-01-16
from sklearn.feature_extraction import DictVectorizer
import unittest


class TestSklearn(unittest.TestCase):

    @staticmethod
    def dictvec():
        """
        字典数据抽取
        :return:None
        """
        # 实力化,指定sparse=False则fit_transform返回sparse的二维矩阵
        dict = DictVectorizer(sparse=False)

        # 调用fit_transform,返回sparse矩阵格式，使用坐标指定数值，sparse矩阵格式节约内存，方便读取；大多数情况不使用这种格式
        data = dict.fit_transform([{'city': 'beijing', 'temperature': 100}, {'city': 'shanghai', 'temperature': 60},
                                   {'city': 'shenzhen', 'temperature': 60}])  # type: object

        print data
        return None

    def test_dictvec(self):
        TestSklearn.dictvec()

# if __name__ == "__main__":
#     dictvec()
