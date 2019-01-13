#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: lo time:2019-01-13


def bubbleSort(arr):
    for i in range(len(arr)-1):
        for j in range(len(arr)-i-1):
            if arr[j]>arr[j+1]:
                arr[j],arr[j+1]=arr[j+1],arr[j]
    return arr


arr=[3,2,6,9,5,1,4,7,8,0]
print bubbleSort(arr)
