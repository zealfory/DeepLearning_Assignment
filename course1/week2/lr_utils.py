# -*- encoding: utf-8 -*-
# @File:   lr_utils.py    
# @Time: 2020-03-13 16:26
# @Author: ZHANG
# @Description: lr_utils

import numpy as np
import h5py


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # [209, 64, 64, 3]
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # 209

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])  # [b'cat' b'noncat']

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))  # [1, 209]
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
