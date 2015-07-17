__author__ = 'du'

import numpy as np
from sklearn import base
from sklearn import preprocessing
from scipy.optimize import minimize
from scipy.misc import logsumexp
from ..utils import sgd


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


class _BaseLogisticRegression(base.BaseEstimator, base.ClassifierMixin):
    def __init__(self, n_dim, n_classes, fit_intercept=True, n_iter=1000, tol=1e-5, C=0.01, **parms):
        self.n_dim = n_dim
        self.n_classes = n_classes
        self.fit_intercept = fit_intercept
        if self.fit_intercept:
            self.W_ = np.random.randn(n_classes, self.n_dim + 1)
        else:
            self.W_ = np.random.randn(n_classes, self.n_dim)
        self.n_iter = n_iter
        self.tol = tol
        self.C = C
        self._preprocessor = preprocessing.LabelBinarizer()

    def fit(self, X, y):
        if self.fit_intercept:
            D = np.column_stack((X, np.ones(X.shape[0])))
        else:
            D = X

        Y = self._preprocessor.fit_transform(y)

        self._fit(D, Y)

        return self

    def _fit(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X):
        if self.fit_intercept:
            D = np.column_stack((X, np.ones(X.shape[0])))
        else:
            D = X
        return softmax(D.dot(self.W_.T))

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


class SGDLogisticRegression(_BaseLogisticRegression):
    def __init__(self, n_dim, n_classes, fit_intercept=True, n_iter=1000, tol=1e-5, C=0.01, batch_size=100, lr=0.001,
                 report=0):
        """
        Logistic Regression using Stocastic Gradient Descent(SGD)
        :param n_dim: int dimension of independent variable
        :param n_classes: int number of classes
        :param fit_intercept: boolean
            if True explicitly fit intercept, if False assume the independent variables include dummy ones
        :param n_iter: int, maximum iteration
        :param tol: float, error tolerance under which the iteration will stop
        :param C: float, L2 regularization parameter
        :param batch_size: int batch size
        :param lr: float learning ratio
        """
        _BaseLogisticRegression.__init__(self, n_dim, n_classes, fit_intercept, n_iter, tol, C)
        self.lr = lr
        self.batch_size = batch_size
        self.report = report

    def _fit(self, X, y):
        # N = len(y)
        # error = 1e10
        # W = self.W_.flatten()
        # for i in range(self.n_iter):
        #     if 0 < self.batch_size < N:
        #         idx = np.random.permutation(self.batch_size)
        #         XX = X[idx]
        #         yy = y[idx]
        #     else:
        #         XX = X
        #         yy = y
        #     _, grad = cross_entropy_error_grad(W, XX, yy, self.C)
        #     W -= self.lr * grad
        #     e = cross_entropy_error(W, X, y, self.C)
        #     if i % 100 == 0:
        #         print(i, e, error - e, self.tol)
        #     if error - e < self.tol:
        #         print(i, e, error - e, self.tol)
        #         break
        #     error = e

        W, error, converged = sgd(cross_entropy_error, cross_entropy_error_grad, self.W_.flatten(), X, y,
                                  self.C, self.n_iter, self.lr, self.batch_size, self.tol, self.report)

        self.W_ = W.reshape(self.n_classes, W.size / self.n_classes)


class BatchLogisticRegression(_BaseLogisticRegression):
    def __init__(self, n_dim, n_classes, fit_intercept=True, n_iter=1000, tol=1e-5, C=0.01, method="CG"):
        """
        Logistic Regression using Batch Gradient Descent such as Conjugate Gradient (CG) or BFGS
        :param n_dim: int dimension of independent variable
        :param fit_intercept: boolean
            if True explicitly fit intercept, if False assume the independent variables include dummy ones
        :param n_iter: int, maximum iteration
        :param tol: float, error tolerance under which the iteration will stop
        :param C: float, L2 regularization parameter
        :param method: string method name passes to scipy.optimize.minimize
        """
        _BaseLogisticRegression.__init__(self, n_dim, n_classes, fit_intercept, n_iter, tol, C)
        self.method = method

    def _fit(self, X, y):
        result = minimize(cross_entropy_error_grad, self.W_.flatten(), (X, y, self.C), self.method, jac=True,
                          tol=self.tol, options={"maxiter": self.n_iter})
        self.W_ = result.x.reshape(self.n_classes, result.x.size / self.n_classes)
