# -*- encoding: utf-8 -*-
# @File:   planar_data_classification_with_one_hidden_layer.py    
# @Time: 2020-03-18 11:23
# @Author: ZHANG
# @Description: planar_data_classification_with_one_hidden_layer

# packages
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from loguru import logger
from course1.week3.planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

logger.add('runtime.log')
np.random.seed(1)

# dataset
X, Y = load_planar_dataset()
# plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)
# plt.show()

# Simple Logistic Regression with sklean.linear_model.LogisticRegression

# clf = sklearn.linear_model.LogisticRegression()
# clf.fit(X.T, Y.T.ravel())
# # plot_decision_boundary(lambda x: clf.predict(x), X, Y)
# # plt.title("Logistic Regression")
# # plt.show()
# LR_predictions = clf.predict(X.T)
# logger.info('Accuracy of logistic regression: %f' % float(
#     (np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions))/float(Y.size))
# )


# Neural Network model
def layer_sizes(X, Y):
    """Defining the neural network structure"""
    n_x = X.shape[0]
    n_y = Y.shape[0]
    return n_x, n_y


def initialize_parameters(n_x, n_h, n_y):
    """Initialize the model's parameters"""
    np.random.seed(2)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    parameters = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }
    return parameters


def forward_propagation(X, parameters):
    """Forward propagation"""
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    Z1 = np.matmul(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.matmul(W2, A1) + b2
    A2 = sigmoid(Z2)
    cache = {
        'Z1': Z1,
        'A1': A1,
        'Z2': Z2,
        'A2': A2
    }
    return A2, cache


def compute_cost(A2, Y, parameters):
    """Computes the cross-entropy cost"""
    m = Y.shape[1]
    logprobs = np.multiply(Y, np.log(A2)) + np.multiply(1 - Y, np.log(1 - A2))
    cost = - 1.0 / m * np.sum(logprobs)
    return cost


def backward_propagation(parameters, cache, X, Y):
    """Backward propagation"""
    m = X.shape[1]

    W1 = parameters['W1']
    W2 = parameters['W2']

    A1 = cache['A1']
    A2 = cache['A2']

    dZ2 = A2 - Y
    dW2 = 1.0 / m * np.matmul(dZ2, A1.T)
    db2 = 1.0 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.matmul(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1.0 / m * np.matmul(dZ1, X.T)
    db1 = 1.0 / m * np.sum(dZ1, axis=1, keepdims=True)


    grads = {
        'dW2': dW2,
        'db2': db2,
        'dW1': dW1,
        'db1': db1

    }
    return grads


def update_parameters(parameters, grads, learning_rate=1.2):
    """Updates parameters using the gradient descent update rule given above"""
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }
    return parameters


def nn_model(X, Y, n_h, num_iterations=10000, learning_rate=1.2, print_cost=False):
    """Neural network model"""
    np.random.seed(3)
    n_x, n_y = layer_sizes(X, Y)
    parameters = initialize_parameters(n_x, n_h, n_y)
    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(cache['A2'], Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i % 1000 == 0:
            logger.info("Cost after iteration %i:%f" % (i, cost))
    return parameters


def predict(parameters, X):
    """Using the learned parameters, predicts a class for each example in X"""
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)
    return predictions


# Test
my_parameters = nn_model(X, Y, n_h=4, num_iterations=10000, learning_rate=0.5, print_cost=True)

plot_decision_boundary(lambda x: predict(my_parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
# plt.show()

predictions = predict(my_parameters, X)
logger.info('Accuracy: %f' %
            float((np.dot(Y, predictions.T)
                   + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size)))
