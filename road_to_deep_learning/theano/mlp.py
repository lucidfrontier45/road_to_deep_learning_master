__author__ = 'du'

import numpy as np
import theano
from theano import tensor as T, function as F, grad as G
from sklearn import base, preprocessing

from ..utils import format_result, log_softmax
from .functions import sigmoid, tanh, identity, relu

floatX = theano.config.floatX


class MLP(base.BaseEstimator):
    def __init__(self, input_dim, hidden_dim, output_dim, activation=relu, lr=0.01, n_iter=100, tol=1e-5, report=100,
                 C_hidden=1.0, C_out=1.0, **params):
        self.Wh = theano.shared(np.random.randn(hidden_dim, input_dim), name="Wh")
        self.bh = theano.shared(np.random.randn(hidden_dim).astype(floatX), name="bh")
        self.Wo = theano.shared(np.random.randn(output_dim, hidden_dim), name="Wo")
        self.bo = theano.shared(np.random.randn(output_dim).astype(floatX), name="bo")
        self.activation = activation
        self.lr_symb = theano.shared(lr, name="lr")
        self.Ch = theano.shared(C_hidden, name="Ch")
        self.Co = theano.shared(C_out, name="Co")
        self.n_iter = n_iter
        self.tol = tol
        self.report = report
        self._forward = self._make_forward_func()

    def _make_forward_func(self):
        x_symb = T.dmatrix("X")
        ah = T.dot(x_symb, self.Wh.T) + self.bh
        zh = self.activation(ah)
        ao = T.dot(zh, self.Wo.T) + self.bo
        return F([x_symb], ao)

    def error(self, y, z):
        raise NotImplementedError

    def grad(self, error):
        grad_Wh = G(error, self.Wh)
        grad_bh = G(error, self.bh)
        grad_Wo = G(error, self.Wo)
        grad_bo = G(error, self.bo)
        return grad_Wh, grad_bh, grad_Wo, grad_bo

    def compile(self, X, y):
        N = len(y)
        x_symb = theano.shared(X, name="X")
        y_symb = theano.shared(y, name="y")
        ah = T.dot(x_symb, self.Wh.T) + self.bh
        zh = self.activation(ah)
        ao = T.dot(zh, self.Wo.T) + self.bo
        error = self.error(y_symb, ao) + self.Co * (self.Wo ** 2).sum() + self.Ch * (self.Wh ** 2).sum()
        grad_Wh, grad_bh, grad_Wo, grad_bo = self.grad(error)
        lr = self.lr_symb / len(y)
        updates = [(self.Wh, self.Wh - grad_Wh * lr), (self.bh, self.bh - grad_bh * lr),
                   (self.Wo, self.Wo - grad_Wo * lr), (self.bo, self.bo - grad_bo * lr)]
        return F([], error, updates=updates)

    def fit(self, X, y):
        f = self.compile(X, y)
        scale_factor = 1.0 / len(y)
        e_old = 1e10
        for epoch in range(self.n_iter):
            e = float(f()) * scale_factor
            de = e_old - e
            if epoch % self.report == 0:
                print(format_result(epoch, e, de, self.tol))
            if de < self.tol:
                print(format_result(epoch, e, de, self.tol))
                print("break")
                break
            e_old = e
        return self

    def get_variables(self):
        Wh = self.Wh.get_value()
        bh = self.bh.get_value()
        Wo = self.Wo.get_value()
        bo = self.bo.get_value()
        return Wh, bh, Wo, bo


class MLPRegressor(MLP, base.RegressorMixin):
    def error(self, y, z):
        y_hat = z
        return ((y - y_hat) ** 2).sum() * 0.5

    def predict(self, X):
        z = self._forward(X)
        return z


class MLPClassifier(MLP, base.ClassifierMixin):
    def error(self, y, z):
        y_hat = T.nnet.softmax(z)
        return T.nnet.categorical_crossentropy(y_hat, y).sum()

    def fit(self, X, y):
        y = preprocessing.LabelBinarizer().fit_transform(y)
        return MLP.fit(self, X, y)

    def predict(self, X):
        z = self._forward(X)
        y_hat = log_softmax(z)
        return y_hat.argmax(1)


class MLPTransformer(MLP, base.TransformerMixin):
    def __init__(self, input_dim, hidden_dim, **params):
        MLP.__init__(self, input_dim, hidden_dim, input_dim, **params)

    def _make_forward_func(self):
        x_symb = T.dmatrix("X")
        ah = T.dot(x_symb, self.Wh.T) + self.bh
        zh = self.activation(ah)
        return F([x_symb], zh)

    def error(self, y, z):
        y_hat = z
        return ((y - y_hat) ** 2).sum() * 0.5

    def fit(self, X, y=None):
        if y is None:
            y = X.copy()

        return MLP.fit(self, X, y)

    def transform(self, X):
        return self._forward(X)

class AutoEncoder(MLPTransformer):
    pass