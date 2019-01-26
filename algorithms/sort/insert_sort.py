#!/usr/bin/env python
# -*- coding:utf-8 -*-
import unittest


class TestInsertSort(unittest.TestCase):
    arr = [3, 2, 6, 9, 5, 1, 4, 7, 8, 0, 10, 11, 12, 13]

    @staticmethod
    def insert_sort(arr):
        for i in range(1, len(arr)):
            tem = arr[i]
            last_index = i
            for j in range(i, 0, -1):
                if tem < arr[j - 1]:
                    arr[j] = arr[j - 1]
                    last_index = j - 1

            arr[last_index] = tem

        return arr

    def test_select(self):
        print(TestInsertSort.arr)
        print(TestInsertSort.insert_sort(TestInsertSort.arr))
