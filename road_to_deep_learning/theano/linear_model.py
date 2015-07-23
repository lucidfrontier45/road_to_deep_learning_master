__author__ = 'du'

import numpy as np
import theano
from theano import tensor as T, function as F, grad as G
from sklearn import base, preprocessing

from ..utils import format_result, log_softmax

floatX = theano.config.floatX


def identity(a):
    return a


class LinearModel(base.BaseEstimator):
    def __init__(self, input_dim, output_dim, lr=0.01, n_iter=100, tol=1e-5, report=100, C=1.0, **params):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.n_iter = n_iter
        self.tol = tol
        self.report = report
        self.C = C

        self.W_symb = theano.shared(np.random.randn(output_dim, input_dim), name="W")
        self.b_symb = theano.shared(np.random.randn(output_dim).astype(floatX), name="b")
        self.lr_symb = theano.shared(lr, name="lr")
        self.C_symb = theano.shared(C, name="C")

    def error(self, y, z):
        raise NotImplementedError

    def grad(self, error):
        grad_W = G(error, self.W_symb)
        grad_b = G(error, self.b_symb)
        return grad_W, grad_b

    def compile(self, X, y):
        N = len(y)
        x_symb = theano.shared(X, name="X")
        y_symb = theano.shared(y, name="y")
        z = T.dot(x_symb, self.W_symb.T) + self.b_symb
        error = self.error(y_symb, z) + self.C_symb * (self.W_symb ** 2).sum()
        grad_W, grad_b = self.grad(error)
        lr = self.lr_symb / len(y)
        updates = [(self.W_symb, self.W_symb - grad_W * lr),
                   (self.b_symb, self.b_symb - grad_b * lr)]
        return F([], error, updates=updates)

    def fit(self, X, y):
        f = self.compile(X, y)
        scale_factor = 1.0 / len(y)
        e_old = 1e10
        for epoch in range(self.n_iter):
            e = float(f()) * scale_factor
            de = e_old - e
            if self.report > 0 and epoch % self.report == 0:
                print(format_result(epoch, e, de, self.tol))
            if de < self.tol:
                if self.report > 0:
                    print(format_result(epoch, e, de, self.tol))
                break
            e_old = e
        return self

    def get_variables(self):
        W = self.W_symb.get_value()
        b = self.b_symb.get_value()
        return W, b

    def _forward(self, X):
        W, b = self.get_variables()
        return np.dot(X, W.T) + b


class LinearRegression(LinearModel, base.RegressorMixin):
    def error(self, y, z):
        y_hat = z
        return ((y - y_hat) ** 2).sum() * 0.5

    def predict(self, X):
        z = self._forward(X)
        return z


class LogisticRegression(LinearModel, base.ClassifierMixin):
    def error(self, y, z):
        y_hat = T.nnet.softmax(z)
        return T.nnet.categorical_crossentropy(y_hat, y).sum()

    def fit(self, X, y):
        y = preprocessing.MultiLabelBinarizer().fit_transform(np.atleast_2d(y).T)
        return LinearModel.fit(self, X, y)

    def predict(self, X):
        z = self._forward(X)
        y_hat = log_softmax(z)
        return y_hat.argmax(1)
