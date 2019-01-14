#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: lo time:2019-01-13
import unittest


class TestSort(unittest.TestCase):
    def tearDown(self):
        # 每个测试用例执行之后做操作
        print('after')

    def setUp(self):
        # 每个测试用例执行之前做操作
        print('before')

    @staticmethod
    def bubbleSort(arr):
        for i in range(len(arr)-1):
            for j in range(len(arr)-i-1):
                if arr[j]>arr[j+1]:
                    arr[j],arr[j+1]=arr[j+1],arr[j]
        return arr

    def test_bubble(self):
        arr=[3,2,6,9,5,1,4,7,8,0]
        print TestSort.bubbleSort(arr)


if __name__ == '__main__':
    unittest.main()

