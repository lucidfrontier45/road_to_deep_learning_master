__author__ = 'du'

import numpy as np
from sklearn import base
from scipy.optimize import minimize

from ..utils import sgd


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


class _BaseLinearRegression(base.BaseEstimator, base.RegressorMixin):
    def __init__(self, n_dim, fit_intercept=True, n_iter=1000, tol=1e-5, C=0.01, **parms):
        self.n_dim = n_dim
        self.fit_intercept = fit_intercept
        if self.fit_intercept:
            self.W_ = np.random.randn(self.n_dim + 1)
        else:
            self.W_ = np.random.randn(self.n_dim)
        self.n_iter = n_iter
        self.tol = tol
        self.C = C

    def fit(self, X, y):
        if self.fit_intercept:
            D = np.column_stack((X, np.ones(X.shape[0])))
        else:
            D = X

        self._fit(D, y)

        return self

    def _fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        if self.fit_intercept:
            D = np.column_stack((X, np.ones(X.shape[0])))
        else:
            D = X

        return D.dot(self.W_)


class SGDLinearRegression(_BaseLinearRegression):
    def __init__(self, n_dim, fit_intercept=True, n_iter=1000, tol=1e-5, C=0.01, batch_size=100, lr=0.001, report=0):
        """
        Linear Regression using Stocastic Gradient Descent(SGD)
        :param n_dim: int dimension of independent variable
        :param fit_intercept: boolean
            if True explicitly fit intercept, if False assume the independent variables include dummy ones
        :param n_iter: int, maximum iteration
        :param tol: float, error tolerance under which the iteration will stop
        :param C: float, L2 regularization parameter
        :param batch_size: int batch size
        :param lr: float learning ratio
        """
        _BaseLinearRegression.__init__(self, n_dim, fit_intercept, n_iter, tol, C)
        self.lr = lr
        self.batch_size = batch_size
        self.report = report

    def _fit(self, X, y):
        W, error, converged = sgd(sum_of_square_error, sum_of_square_error_grad, self.W_.copy(), X, y,
                                  self.C, self.n_iter, self.lr, self.batch_size, self.tol, self.report)

        self.W_ = W.copy()


class BatchLinearRegression(_BaseLinearRegression):
    def __init__(self, n_dim, fit_intercept=True, n_iter=1000, tol=1e-5, C=0.01, method="CG"):
        """
        Linear Regression using Batch Gradient Descent such as Conjugate Gradient (CG) or BFGS
        :param n_dim: int dimension of independent variable
        :param fit_intercept: boolean
            if True explicitly fit intercept, if False assume the independent variables include dummy ones
        :param n_iter: int, maximum iteration
        :param tol: float, error tolerance under which the iteration will stop
        :param C: float, L2 regularization parameter
        :param method: string method name passes to scipy.optimize.minimize
        """
        _BaseLinearRegression.__init__(self, n_dim, fit_intercept, n_iter, tol, C)
        self.method = method

    def _fit(self, X, y):
        result = minimize(sum_of_square_error_grad, self.W_, (X, y, self.C), self.method, jac=True,
                          tol=self.tol,
                          options={"maxiter": self.n_iter})
        self.W_ = result.x


if __name__ == "__main__":
    np.random.seed(0)
    W = [2, -1]
    X = np.random.randn(10000, 2)
    y = X.dot(W) + 1 + np.random.randn(len(X))

    model1 = SGDLinearRegression(X.shape[1], tol=1e-7, batch_size=100, n_iter=1000000).fit(X, y)
    print(model1.W_)
    print(model1.score(X, y))

    model2 = BatchLinearRegression(X.shape[1], tol=1e-7, n_iter=1000000, method="L-BFGS-B").fit(X, y)
    print(model2.W_)
    print(model2.score(X, y))
