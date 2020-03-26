# -*- encoding: utf-8 -*-
# @File:   build_dnn_step_by_step.py    
# @Time: 2020-03-25 15:14
# @Author: ZHANG
# @Description: build_dnn_step_by_step

import numpy as np
import h5py
import matplotlib.pyplot as plt
from course1.week4.dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward


plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)


# Initialization
# def initialize_parameters(n_x, n_h, n_y):
#     """Create and initialize the parameters of the 2-layer neural network"""
#     # todo
#     pass


def initialize_parameters_deep(layer_dims):
    """Implement initialization for an L-layer Neural Network"""
    # todo
    pass


# Forward propagation
def linear_forward(A, W, b):
    """Implement the linear part of a layer's forward propagation"""
    # todo
    pass


def linear_activation_forward(A_prev, W, b, activation):
    """Implement the forward propagation for the LINEAR->ACTIVATION layer"""
    # todo
    pass


def L_model_forward(X, parameters):  # L-Layer Model
    """Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation"""
    # todo
    pass


# Cost function
def compute_cost(AL, Y):
    """Implement the cost function"""
    # todo
    pass


# Backward propagation
def linear_backward(dZ, cache):
    """Implement the linear portion of backward propagation for a single layer (layer l)"""
    # todo
    pass


def linear_activtion_backward(dA, cache, activation):
    """Implement the backward propagation for the LINEAR->ACTIVATION layer"""
    # todo
    pass


def L_model_backward(AL, Y, caches):  # L-Model Backward
    """Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group"""
    # todo
    pass


# Update Parameters
def update_parameters(parameters, grads, learning_rate):
    """Update parameters using gradient decent"""
    # todo
    pass