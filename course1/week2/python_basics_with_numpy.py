# -*- encoding: utf-8 -*-
# @File:   python_basics_with_numpy.py    
# @Time: 2020-03-12 14:00
# @Author: ZHANG
# @Description: python_basics_with_numpy

test = "Hello World"
print('test:', test)


import math


def basic_sigmoid(x):
    s = 1.0 / (1 + math.exp(-x))
    return s


print('basic sigmoid:', basic_sigmoid(3))

# x = [1, 2, 3]
# print(basic_sigmoid(x))  # error

import numpy as np
x = np.array([1, 2, 3])
print('numpy exp:\n', np.exp(x))

print('x + 3:\n', (x + 3))
