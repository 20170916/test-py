#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: lo time:2019-01-13
import unittest


class TestSwapSort(unittest.TestCase):
    arr = [3, 2, 6, 9, 5, 1, 4, 7, 8, 0, 10, 11, 12, 13]

    def tearDown(self):
        # 每个测试用例执行之后做操作
        print('after')

    def setUp(self):
        # 每个测试用例执行之前做操作
        print('before')

    @staticmethod
    def swap(arr, i, j):
        arr[i] ^= arr[j]
        arr[j] ^= arr[i]
        arr[i] ^= arr[j]
        return arr

    def test_swap(self):
        print(TestSwapSort.arr)
        print(TestSwapSort.swap(TestSwapSort.arr, 0, 1))

    @staticmethod
    def bubbleSort(arr):
        for i in range(len(arr) - 1):
            for j in range(len(arr) - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr

    def test_bubble(self):
        print(TestSwapSort.bubbleSort(TestSwapSort.arr))

    @staticmethod
    def optimize_bubbleSort(arr):
        # for i in range(len(arr) - 1, 0, -1):
        i = len(arr) - 1
        while i > 0:
            last_swap_index = 0
            for j in range(i):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    last_swap_index = j
            i = last_swap_index
        return arr

    def test_optimize_bubbleSort(self):
        print(TestSwapSort.arr)
        print(TestSwapSort.optimize_bubbleSort(TestSwapSort.arr))

    @staticmethod
    def quick_sort(arr):
        TestSwapSort.quick_sort_rec(arr, 0, len(arr) - 1)
        return arr

    @staticmethod
    def med3(arr, a, b, c):
        if arr[a] >= arr[b]:
            if arr[b] >= arr[c]:
                return arr[b]
            if arr[c] >= arr[a]:
                return arr[a]
            return arr[c]
        if arr[a] <= arr[b]:
            if arr[c] <= arr[a]:
                return arr[a]
            if arr[b] <= arr[c]:
                return arr[b]
            return arr[c]

    @staticmethod
    def quick_sort_rec(arr, s, e):
        if s + 1 == e and arr[s] > arr[e]:
            TestSwapSort.swap(arr, s, e)
            return
        if s < e:
            # 此时至少3个数
            mid = (e + s) // 2
            pivot_value = TestSwapSort.med3(arr, e, s, mid)
            i = s
            j = e
            while i <= j:
                while i <= j and arr[i] <= pivot_value:
                    i += 1
                # 此时可保证i左侧都是小于等于枢轴值的，不包括i。
                # 此时i若越界，则所有值小于枢轴值，
                # 此时i若不越界，i指向的值一定大于枢轴值；
                # j如果能走到i的位置，那j一定还能再走一步，最终j==i-1；左[s,j],右[i,e]
                while i <= j and arr[j] >= pivot_value:
                    j -= 1
                # 此时j要么与i相遇后又走了一步，要么没与i相遇
                # 若i与j相遇，则j==i-1；即左[s,j],右[i,e]
                # 若j与i没相遇，交换i，j的值，并各自向前走一步
                if i < j:
                    TestSwapSort.swap(arr, i, j)
                    if j-i > 1:
                        i += 1
                        j -= 1

            TestSwapSort.quick_sort_rec(arr, s, j)
            TestSwapSort.quick_sort_rec(arr, i, e)
            return
        return

    def test_quick_sort(self):
        arr = TestSwapSort.arr
        print(arr)
        print('quick sort\n', TestSwapSort.quick_sort(arr))

# if __name__ == '__main__':
#     unittest.main()
