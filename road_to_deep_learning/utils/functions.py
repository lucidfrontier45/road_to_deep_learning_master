__author__ = 'du'

import numpy as np
from scipy.misc import logsumexp


def sum_of_square_error(W, X, y, C=0):
    """
    sum of square cost function for linear regression
    :param W: ndarray Coefficients
    :param X: ndarray Independent variable
    :param y: ndarray Dependent variable
    :param C: float parameter for L2 regularization
    :return: value of cost function
    """
    e = - (y - X.dot(W))
    error = e.dot(e)

    if C > 0:
        error += C * W.dot(W)

    return error


def sum_of_square_error_grad(W, X, y, C=0):
    """
    sum of square cost function and its gradient for linear regression
    :param W: ndarray Coefficients
    :param X: ndarray Independent variable
    :param y: ndarray Dependent variable
    :param C: float parameter for L2 regularization
    :return: tuple of cost function and gradient
    """
    e = - (y - X.dot(W))
    error = e.dot(e)
    grad = np.sum(e[:, np.newaxis] * X, 0)

    if C > 0:
        error += C * W.dot(W)
        grad += 2.0 * C * W

    return error, grad


def softmax(w):
    w = np.array(w)
    maxes = np.max(w, axis=1)
    maxes = maxes.reshape(maxes.shape[0], 1)
    e = np.exp(w - maxes)
    dist = e / np.sum(e, axis=1)[:, np.newaxis]

    return dist


def log_softmax(w):
    w = np.array(w)
    maxes = np.max(w, axis=1)
    maxes = maxes.reshape(maxes.shape[0], 1)
    e = w - maxes
    return e - logsumexp(e, axis=1)[:, np.newaxis]


def cross_entropy_error(W, X, y, C=0):
    n_dim = X.shape[1]
    n_classes = W.size / n_dim
    W = W.reshape(n_classes, n_dim)
    ln_h = log_softmax(X.dot(W.T))
    error = -(y * ln_h).sum()

    if C > 0:
        error += C * W.dot(W.T).sum()

    return error


def cross_entropy_error_grad(W, X, y, C=0):
    """
    cross-entropy cost function and its gradient for logistic regression
    :param W: ndarray, shape(K, D) Coefficients
    :param X: ndarray, shape(N, D) Independent variable
    :param y: ndarray, shape(N, K) Dependent variable in 1-of-K encoding
    :param C: float parameter for L2 regularization
    :return: tuple of cost function and gradient
    """

    n_dim = X.shape[1]
    n_classes = W.size / n_dim
    W = W.reshape(n_classes, n_dim)
    ln_h = log_softmax(X.dot(W.T))
    error = -(y * ln_h).sum()
    grad = (np.exp(ln_h) - y).T.dot(X)

    if C > 0:
        error += C * W.dot(W.T).sum()
        grad += 2.0 * C * W

    return error, grad.flatten()
