__author__ = 'shiqiaodu'

import numpy as np
from sklearn import base, preprocessing
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

    delta_hidden = (y_hat - y)
    delta_input = delta_hidden.dot(w_hidden.T) * f_grad(a)
    grad_hidden = delta_hidden.T.dot(z)
    grad_input = delta_input.T.dot(x)
    return grad_input, grad_hidden.T


def _error_func(w, x, y, f, lambda_input, lambda_hidden, typ="regression"):
    w_input, w_hidden = _unpack_w(w, x, y)
    a, z, y_hat = _forward(w_input, w_hidden, x, f)
    if typ == "regression":
        e = np.sum((y - y_hat) ** 2) * 0.5
    elif typ == "classification":
        ln_h = functions.log_softmax(y_hat)
        e = -(y * ln_h).sum()
    else:
        raise ValueError("unknown typ {0}".format(typ))
    if lambda_input > 0:
        e += lambda_input * (w_input ** 2).sum()
    if lambda_hidden > 0:
        e += lambda_hidden * (w_hidden ** 2).sum()
    return e


def _error_grad_func(w, x, y, f, f_grad, lambda_input, lambda_hidden, typ="regression"):
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
    if typ == "regression":
        e = np.sum((y - y_hat) ** 2) * 0.5
    elif typ == "classification":
        ln_h = functions.log_softmax(y_hat)
        e = -(y * ln_h).sum()
        np.exp(ln_h, out=y_hat)
    else:
        raise ValueError("unknown typ {0}".format(typ))
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


class MLP(base.BaseEstimator):
    def __init__(self, input_dim, hidden_dim, output_dim, typ, activation=functions.Sigmoid(),
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
                                   lambda_hidden=lambda_hidden, typ=typ)
        self._error_grad_func = partial(_error_grad_func, f=self.activation, f_grad=self.activation.grad,
                                        lambda_input=lambda_input, lambda_hidden=lambda_hidden, typ=typ)

    def _forward(self, X):
        return _forward(self.w_input, self.w_hidden, X, self.activation)

    def forward(self, X):
        if self.fit_intercept:
            D = np.column_stack((X, np.ones(X.shape[0])))
        else:
            D = X
        return _forward(self.w_input, self.w_hidden, D, self.activation)

    def _fit(self, X, y):
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


class MLPRegressor(MLP, base.RegressorMixin):
    def __init__(self, input_dim, hidden_dim, output_dim, **params):
        MLP.__init__(self, input_dim, hidden_dim, output_dim, "regression", **params)

    def fit(self, X, y):
        return self._fit(X, y)

    def predict(self, X):
        a, z, y_hat = self.forward(X)
        return y_hat


class MLPClassifier(MLP, base.ClassifierMixin):
    def __init__(self, input_dim, hidden_dim, output_dim, **params):
        MLP.__init__(self, input_dim, hidden_dim, output_dim, "classification", **params)

    def fit(self, X, y):
        y = preprocessing.LabelBinarizer().fit_transform(y)
        return self._fit(X, y)

    def predict_proba(self, X):
        a, z, y_hat = self.forward(X)
        p = np.exp(functions.log_softmax(y_hat))
        return p

    def predict(self, X):
        a, z, y_hat = self.forward(X)
        ln_h = functions.log_softmax(y_hat)
        return ln_h.argmax(1)


class MLPTransformer(MLP, base.TransformerMixin, base.RegressorMixin):
    def __init__(self, input_dim, hidden_dim, **params):
        MLP.__init__(self, input_dim, hidden_dim, input_dim, "regression", fit_intercept=False, **params)

    def fit(self, X, y=None):
        return self._fit(X, X)

    def transform(self, X):
        a, z, y_hat = self.forward(X)
        return z

    def predict(self, X):
        a, z, y_hat = self.forward(X)
        return y_hat


class AutoEncoder(MLPTransformer):
    pass


class DenoisingAutoEncoder(MLPTransformer):
    def __init__(self, input_dim, hidden_dim, noise_ratio=0.1, **params):
        MLP.__init__(self, input_dim, hidden_dim, input_dim, "regression", fit_intercept=False, **params)
        self.noise_ratio = noise_ratio

    def fit(self, X, y=None):
        noisy_X = X + np.random.randn(*X.shape) * X.std(0)[np.newaxis, :] * self.noise_ratio
        return self._fit(noisy_X, X)
