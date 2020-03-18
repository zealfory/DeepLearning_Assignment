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
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

logger.add('runtime.log')
np.random.seed(1)

# dataset
X, Y = load_planar_dataset()
# plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)
# plt.show()

# Simple Logistic Regression
clf = sklearn.linear_model.LogisticRegression()
clf.fit(X.T, Y.T.ravel())
# plot_decision_boundary(lambda x: clf.predict(x), X, Y)
# plt.title("Logistic Regression")
# plt.show()
LR_predictions = clf.predict(X.T)
logger.info('Accuracy of logistic regression: %s %% percentage of correctly labelled datapoints' % float(
    (np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions))/float(Y.size)*100)
)


# Neural Network model
def layer_sizes(X, Y):
    """Defining the neural network structure"""
    # TODO
    pass


def initialize_parameters(n_x, n_h, n_y):
    """Initialize the model's parameters"""
    # TODO
    pass


def forward_propagation(X, parameters):
    """Forward propagation"""
    # TODO
    pass


def compute_cost(A2, Y, parameters):
    """Computes the cross-entropy cost"""
    # todo
    pass


def backward_propagation(parameters, cache, X, Y):
    """Backward propagation"""
    # todo
    pass


def update_parameters(parameters, grads, learning_rate = 1.2):
    """Updates parameters using the gradient descent update rule given above"""
    # todo
    pass


def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    """Neural network model"""
    # todo
    pass


def predict(parameters, X):
    """Using the learned parameters, predicts a class for each example in X"""
    # todo
    pass
