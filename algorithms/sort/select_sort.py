#!/usr/bin/env python
# -*- coding:utf-8 -*-
import unittest


class TestSelectSort(unittest.TestCase):
    arr = [3, 2, 6, 9, 5, 1, 4, 7, 8, 0, 10, 11, 12, 13]

    @staticmethod
    def swap(arr, i, j):
        arr[i] ^= arr[j]
        arr[j] ^= arr[i]
        arr[i] ^= arr[j]
        return arr

    @staticmethod
    def select_sort(arr):
        for i in range(len(arr)-1):
            min_index = i
            for j in range(i, len(arr)):
                if arr[j] < arr[min_index]:
                    min_index = j
            if min_index != i:
                TestSelectSort.swap(arr, i, min_index)
        return arr

    def test_select(self):
        print(TestSelectSort.arr)
        print(TestSelectSort.select_sort(TestSelectSort.arr))
