__author__ = 'shiqiaodu'

import numpy as np
from sklearn import base
from functools import partial

from ..utils import optimize, functions


def _pack_w(w_input, w_output):
    """

    :rtype : ndarray
    """
    return np.r_[w_input.flatten(), w_output.flatten()]


def _unpack_w(w, x, y):
    """
    :param w: ndarray, shape(hidden_dim, input_dim + output_dim)
    :param x: ndarray, shape(n_obs, input_dim)
    :param y: ndarray, shape(n_obs, output_dim)
    :return: a[n_obs, n_hidden], z[n_obs, n_hidden], y_hat[n_obs, n_output]
    """

    n_obs, input_dim = x.shape
    _, output_dim = y.shape
    hidden_dim = w.size / (input_dim + output_dim)
    w_input = w[:hidden_dim * input_dim].reshape(hidden_dim, input_dim)
    w_hidden = w[hidden_dim * input_dim:].reshape(hidden_dim, output_dim)
    return w_input, w_hidden


def _forward(w_input, w_hidden, x, f):
    """
    :param w_input: ndarray, shape(hidden_dim, input_dim)
    :param w_hidden: ndarray, shape(hidden_dim, output_dim)
    :param x: ndarray, shape(n_obs, input_dim)
    :param y: ndarray, shape(n_obs, output_dim)
    :param f: function, activation function
    :return: a[n_obs, n_hidden], z[n_obs, n_hidden], y_hat[n_obs, n_output]
    """

    a = x.dot(w_input.T)
    z = f(a)
    y_hat = z.dot(w_hidden)
    return a, z, y_hat


def _backward(w_hidden, x, y, f_grad, a, z, y_hat):
    """
    :param w_hidden: ndarray, shape(hidden_dim, output_dim)
    :param x: ndarray, shape(n_obs, input_dim)
    :param y: ndarray, shape(n_obs, output_dim)
    :param f_grad: function, gradient of activation function
    :param a: ndarray, shape(n_obs, hidden_dim), activations of hidden units
    :param z: ndarray, shape(n_obs, hidden_dim), output of hidden units
    :param y_hat: ndarray, shape(n_obs, output_dim) output of output units
    :return: grad_input[hidden_dim, input_dim], grad_hidden[hidden_dim, output_dim]
    """

    delta_hidden, delta_input = _compute_delta(w_hidden, y, f_grad, a, y_hat)
    grad_hidden = delta_hidden.T.dot(z)
    grad_input = delta_input.T.dot(x)
    return grad_input, grad_hidden.T


def _compute_delta(w_hidden, y, f_grad, a, y_hat):
    """
    :param w_hidden: ndarray, shape(hidden_dim, output_dim)
    :param y: ndarray, shape(n_obs, output_dim)
    :param f_grad: function, gradient of activation function
    :param a: ndarray, shape(n_obs, hidden_dim), activations of hidden units
    :param y_hat: ndarray, shape(n_obs, output_dim) output of output units
    :return: grad_input[hidden_dim, input_dim], grad_hidden[hidden_dim, output_dim]
    """

    delta_hidden = (y_hat - y)
    delta_input = delta_hidden.dot(w_hidden.T) * f_grad(a)
    return delta_hidden, delta_input


def _error_func(w, x, y, f, lambda_input, lambda_hidden):
    w_input, w_hidden = _unpack_w(w, x, y)
    a, z, y_hat = _forward(w_input, w_hidden, x, f)
    e = np.sum((y - y_hat) ** 2) * 0.5
    if lambda_input > 0:
        e += lambda_input * (w_input ** 2).sum()
    if lambda_hidden > 0:
        e += lambda_hidden * (w_hidden ** 2).sum()
    return e


def _error_grad_func(w, x, y, f, f_grad, lambda_input, lambda_hidden):
    """
    :param w: ndarray, shape(hidden_dim, input_dim + output_dim)
    :param x: ndarray, shape(n_obs, input_dim)
    :param y: ndarray, shape(n_obs, output_dim)
    :param f: function, activation function
    :param f_grad: function, gradient of activation function
    :return:
    """

    w_input, w_hidden = _unpack_w(w, x, y)
    a, z, y_hat = _forward(w_input, w_hidden, x, f)
    e = np.sum((y - y_hat) ** 2) * 0.5
    grad_input, grad_hidden = _backward(w_hidden, x, y, f_grad, a, z, y_hat)

    # print(w_input.shape, grad_input.shape)
    # print(w_hidden.shape, grad_hidden.shape)

    if lambda_input > 0:
        e += lambda_input * (w_input ** 2).sum()
        grad_input += 2.0 * lambda_input * w_input
    if lambda_hidden > 0:
        e += lambda_hidden * (w_hidden ** 2).sum()
        grad_hidden += 2.0 * lambda_hidden * w_hidden

    return e, _pack_w(grad_input, grad_hidden)


class MLPRegression(base.BaseEstimator, base.RegressorMixin):
    def __init__(self, input_dim, hidden_dim, output_dim, activation=functions.Sigmoid(),
                 fit_intercept=True, n_iter=1000, tol=1e-5, optimizer=optimize.sgd,
                 batch_size=100, lr=0.001, report=0, momentum=0.9, method="CG",
                 lambda_input=1.0, lambda_hidden=1.0, **params):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.fit_intercept = fit_intercept
        if self.fit_intercept:
            self.w_input = np.random.randn(hidden_dim, input_dim + 1)
        else:
            self.w_input = np.random.randn(hidden_dim, input_dim)
        self.w_hidden = np.random.randn(hidden_dim, output_dim)
        self.n_iter = n_iter
        self.tol = tol
        self.lr = lr
        self.batch_size = batch_size
        self.report = report
        self.momentum = momentum
        self.method = method
        self.optimizer = optimizer
        self.lambda_input = lambda_input
        self.lambda_hidden = lambda_hidden

        self._error_func = partial(_error_func, f=self.activation, lambda_input=lambda_input,
                                   lambda_hidden=lambda_hidden)
        self._error_grad_func = partial(_error_grad_func, f=self.activation, f_grad=self.activation.grad,
                                        lambda_input=lambda_input, lambda_hidden=lambda_hidden)

    def _forward(self, X):
        return _forward(self.w_input, self.w_hidden, X, self.activation)

    def forward(self, X):
        if self.fit_intercept:
            D = np.column_stack((X, np.ones(X.shape[0])))
        else:
            D = X
        return _forward(self.w_input, self.w_hidden, D, self.activation)

    def fit(self, X, y):
        if self.fit_intercept:
            D = np.column_stack((X, np.ones(X.shape[0])))
        else:
            D = X
        w = _pack_w(self.w_input, self.w_hidden)
        w_res, error, success = self.optimizer(self._error_func, self._error_grad_func, w, D, y,
                                               self.n_iter, tol=self.tol, lr=self.lr, batch_size=self.batch_size,
                                               report=self.report, momentum=self.momentum, method=self.method)
        w_input, w_hidden = _unpack_w(w_res, D, y)
        self.w_input = w_input
        self.w_hidden = w_hidden
        return self

    def predict(self, X):
        if self.fit_intercept:
            D = np.column_stack((X, np.ones(X.shape[0])))
        else:
            D = X
        a, z, y_hat = self._forward(D)
        print(y_hat.shape)
        return y_hat
