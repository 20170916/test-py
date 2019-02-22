#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: lo time:2019-02-22
import unittest
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split


class TitanicDecision(unittest.TestCase):

    @staticmethod
    def decision():
        """
        决策树对泰坦尼克号进行预测生死
        :return:
        """

        # 获取数据
        titan = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")

        # 处理数据，找出特征值和目标值
        x = titan[['pclass', 'age', 'sex']]
        y = titan['survived']
        print(x)

        # 处理缺失值
        x['age'].fillna(x['age'].mean(), inplace=True)

        #  分割数据集
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

        # 进行特征工程
        dict = DictVectorizer(sparse=False)
        # 一行特征转化成字典
        x_train = dict.fit_transform(x_train.to_dict(orient="records"))
        print(dict.get_feature_names())
        # 对测试集进行特征抽取
        x_test = dict.transform(x_test.to_dict(orient='records'))
        print(x_train)

        return None

    def test_decision(self):
        TitanicDecision.decision()
