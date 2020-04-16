# -*- encoding: utf-8 -*-
# @File:   dnn_app_utils_v2.py    
# @Time: 2020-03-27 14:01
# @Author: ZHANG
# @Description: dnn_app_utils_v2

import numpy as np
import matplotlib.pyplot as plt
import h5py


def sigmoid(Z):
    """Implement the sigmoid activation in numpy"""
    A = 1.0 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def relu(Z):
    """Implement the RELU function"""
    A = np.maximum(0, Z)
    cache = Z
    return A, cache


def relu_backward(dA, cache):
    """Implement the backward propagation for a single RELU unit"""
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def sigmoid_backward(dA, cache):
    """Implement the backward propagation for a single SIGMOID unit"""
    Z = cache
    s = 1.0 / (1.0 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ


def load_data():
    train_dataset = h5py.File('../week2/datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # (209, 64, 64, 3)
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # (209,)

    test_dataset = h5py.File('../week2/datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # (50, 64, 64, 3)
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # (50,)
    classes = np.array(test_dataset["list_classes"][:])  # (b'cat' b'noncat')
    print('classes:', classes.shape)
    print('-----------------------')
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))  # (1, 209)
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))  # (1, 50)

    print('train_set:', train_set_x_orig.shape, train_set_y_orig.shape)
    print('test_set:', test_set_x_orig.shape, test_set_y_orig.shape)
    print('-----------------------')

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def initialize_parameters(n_x, n_h, n_y):  # n_x = 64x64x3, n_h = ?, n_y = 1
    np.random.seed(1)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {
                  'W1': W1,  # (n_h, 64x64x3)
                  'b1': b1,  # (n_h, 1)
                  'W2': W2,  # (1, n_h)
                  'b2': b2   # (n_y, 1)
                  }
    return parameters


def initialize_parameters_deep(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    return parameters


def linear_forward(A, W, b):
    """Implement the linear part of a layer's forward propagation."""
    Z = np.matmul(W, A) + b

    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """Implement the forwards propagation for the LINEAR->ACTIVATION layer"""
    pass


def L_model_forward(X, parameters):
    """Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation"""
    pass


def compute_cost(AL, Y):
    """Implement the cost function"""
    pass


def linear_backward(dZ, cache):
    """Implement the linear portion of backward propagation for a single layer (layer l)"""
    pass


def linear_activation_backward(dA, cache, activation):
    """Implement the backward propagation for the LINEAR->ACTIVATION layer"""
    pass


def L_model_backward(AL, Y, caches):
    """Implement the backward propagation for the [LINEAR->RELU * (L-1) -> LINEAR -> SIGMOID group"""
    pass


def update_parameters(parameters, grads, learning_rate):
    """Update parameters using gradient descent"""
    pass


def predict(X, y, parameters):
    """Predict the results of a L-layer neural network"""
    pass


def print_mislabeled_images(classes, X, y, p):
    """Plots images where predictions and truth were different"""
    pass


if __name__ == "__main__":
    load_data()


