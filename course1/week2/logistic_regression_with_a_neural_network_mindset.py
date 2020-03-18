# -*- encoding: utf-8 -*-
# @File:   logistic_regression_with_a_neural_network_mindset.py    
# @Time: 2020-03-13 16:20
# @Author: ZHANG
# @Description: logistic_regression_with_a_neural_network_mindset
import numpy as np
from lr_utils import load_dataset
from loguru import logger

logger.add('runtime.log')


def sigmoid(z):
    s = 1.0 / (1 + np.exp(-z))
    return s


def initialize_with_zeros(dim):
    w = np.zeros([dim, 1])  # w: [height x width x channel, 1]
    # w = np.random.normal(size=(dim, 1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))

    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]
    # A:[1, m]
    A = sigmoid(np.dot(w.T, X) + b)  # w.T:[1, height x width x channel] * X:[height x width x channel, m]
    cost = - 1.0 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))  # Y:[1, m]

    # dw:[height x width x channel, 1]
    dw = 1.0 / m * np.dot(X, (A - Y).T)  # X:[height x width x channel, m] * (A - Y).T: [m, 1]
    db = 1.0 / m * np.sum(A - Y)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    # cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw": dw,
             "db": db}
    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    dw = 0.
    db = 0.
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            logger.info("Cost after iteration %i: %f" % (i, cost))
    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}
    return params, grads, costs


def predict(w, b, X):
    m = X.shape[1]
    # Y_prediction = np.zeros((1, m))
    # w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)
    Y_prediction = np.where(A > 0.5, 1, 0)
    assert(Y_prediction.shape == (1, m))
    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=1000, learning_rate=0.5, print_cost=True):
    w, b = initialize_with_zeros(X_train.shape[0])  # X_train:[height x width x channel, m]
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost=print_cost)
    w = parameters["w"]
    b = parameters["b"]
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    logger.info("train accuracy: %.6f" % (1 - np.mean(np.abs(Y_prediction_train - Y_train))))
    logger.info("test accuracy: %.6f" % (1 - np.mean(np.abs(Y_prediction_test - Y_test))))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    return d


# x:[m, height, width, channel] y:[1, m] classes:
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
# x_flatten: [height x width x channel, m]
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)

