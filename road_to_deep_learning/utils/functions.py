__author__ = 'du'

import numpy as np
from scipy.misc import logsumexp
from functools import partial


def l2_penalty(W, C):
    return C * W.dot(W.T).sum()


def l2_penalty_grad(W, C):
    e = l2_penalty(W, C)
    g = 2.0 * C * W
    return e, g


def create_cost_func(cost_f, penalty_f, C, fit_intercept):
    def f(W, X, y, C, fit_intercept):
        W = W.flatten()
        error = cost_f(W, X, y)
        if C > 0:
            if fit_intercept:
                n_dim = X.shape[1]
                W[n_dim - 1::n_dim] = 0
            e = penalty_f(W, C)
            error += e
        return error

    return partial(f, C=C, fit_intercept=fit_intercept)


def create_cost_grad_func(cost_grad_f, penalty_grad_f, C, fit_intercept):
    def f(W, X, y, C, fit_intercept):
        W = W.flatten()
        error, grad = cost_grad_f(W, X, y)
        if C > 0:
            if fit_intercept:
                n_dim = X.shape[1]
                W[n_dim - 1::n_dim] = 0
            e, g = penalty_grad_f(W, C)
            error += e
            grad += g
        return error, grad

    return partial(f, C=C, fit_intercept=fit_intercept)


def sum_of_square_error(W, X, y):
    """
    sum of square cost function for linear regression
    :param W: ndarray Coefficients
    :param X: ndarray Independent variable
    :param y: ndarray Dependent variable
    :return: value of cost function
    """
    e = - (y - X.dot(W))
    error = e.dot(e)

    return error


def sum_of_square_error_grad(W, X, y):
    """
    sum of square cost function and its gradient for linear regression
    :param W: ndarray Coefficients
    :param X: ndarray Independent variable
    :param y: ndarray Dependent variable
    :return: tuple of cost function and gradient
    """
    e = - (y - X.dot(W))
    error = e.dot(e)
    grad = np.sum(e[:, np.newaxis] * X, 0)

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


def cross_entropy_error(W, X, y):
    n_dim = X.shape[1]
    n_classes = W.size / n_dim
    W = W.reshape(n_classes, n_dim)
    ln_h = log_softmax(X.dot(W.T))
    error = -(y * ln_h).sum()

    return error


def cross_entropy_error_grad(W, X, y):
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

    return error, grad.flatten()
