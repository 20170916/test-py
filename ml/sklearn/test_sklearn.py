#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: lo time:2019-01-16
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import unittest
import jieba


class TestSklearn(unittest.TestCase):

    @staticmethod
    def dictvec():
        """
        字典数据抽取
        :return:None
        """
        # 实例化,sparse参数默认为True；指定sparse=False则fit_transform返回sparse的二维矩阵（ndarray）
        dict = DictVectorizer(sparse=False)

        # 调用fit_transform,返回sparse矩阵格式，使用坐标指定数值，sparse矩阵格式节约内存，方便读取；大多数情况不使用这种格式
        data = dict.fit_transform([{'city': 'beijing', 'temperature': 100}, {'city': 'shanghai', 'temperature': 60},
                                   {'city': 'shenzhen', 'temperature': 60}])  # type: object

        print(dict.get_feature_names())

        print(dict.inverse_transform(data))

        print(data)
        return None

    def test_dictvec(self):
        TestSklearn.dictvec()

    @staticmethod
    def countvec():
        """
        对文本进行特征值化
        :return: None
        """
        cv = CountVectorizer()

        # data = cv.fit_transform(["i am a happy man", "he must've had on some really nice pants"])
        # data = cv.fit_transform(["life is is short,i like python", "life is too long,i dislike python"])
        data = cv.fit_transform(["人生苦短，我爱python", "长路慢慢，我爱java"])
        #print data
        # 统计所有文章中没有重复出现的所有词，重复的只看作一次（单个字母不统计，例如i），即【词的列表】
        print(cv.get_feature_names())
        # toarray()方法能将sparse矩阵转化成数组形式，对应每篇文章，在【词的列表】里每个词出现的次数
        print(data.toarray())


    def test_countvec(self):
        TestSklearn.countvec()

    @staticmethod
    def cutword():
        s1="python是世界上最美的语言"
        s2="湖人总冠军"
        s3="你好香啊"
        con1=jieba.cut(s1)
        con2=jieba.cut(s2)
        con3=jieba.cut(s3)

        #转换成列表
        content1=list(con1)
        content2=list(con2)
        content3=list(con3)

        #列表转字符串
        c1=' '.join(content1)
        c2=' '.join(content2)
        c3=' '.join(content3)

        return c1,c2,c3

    @staticmethod
    def chinese_vec():
        """
        汉子特征值化
        :return: None
        """
        c1,c2,c3=TestSklearn.cutword()
        print(c1,c2,c3)
        cv = CountVectorizer()
        data = cv.fit_transform([c1,c2,c3])
        print(cv.get_feature_names())
        print(data.toarray())

    def test_chinese_vec(self):
        TestSklearn.chinese_vec()

# if __name__ == "__main__":
#     dictvec()
