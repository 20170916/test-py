#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: lo time:2019-01-13
import unittest


class TestSort(unittest.TestCase):
    arr = [3, 2, 6, 9, 5, 1, 4, 7, 8, 0]
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
        print TestSort.bubbleSort(TestSort.arr)

    @staticmethod
    def optimize_bubbleSort(arr):
        for i in range(len(arr)-1,0,-1):
            last_swap_index=0;
            for j in range(i):
                if arr[j]>arr[j+1]:
                    arr[j],arr[j+1]=arr[j+1],arr[j]
                    last_swap_index=j
            i=last_swap_index
        return arr

    def test_optimize_bubbleSort(self):
        print TestSort.optimize_bubbleSort(TestSort.arr)


if __name__ == '__main__':
    unittest.main()

