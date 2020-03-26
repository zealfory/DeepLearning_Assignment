# -*- encoding: utf-8 -*-
# @File:   dnn_utils_v2.py    
# @Time: 2020-03-25 15:17
# @Author: ZHANG
# @Description: dnn_utils_v2

import numpy as np


def sigmoid(Z):
    A = 1.0 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache


def sigmoid_backward(dA, cache):
    Z = cache
    s = 1.0 / (1 + np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ