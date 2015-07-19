__author__ = 'du'

import numpy as np
from sklearn import base
from scipy.optimize import minimize

from ..utils import sum_of_square_error, sum_of_square_error_grad, l2_penalty, l2_penalty_grad, create_cost_func, \
    create_cost_grad_func, sgd


class _BaseLinearRegression(base.BaseEstimator, base.RegressorMixin):
    def __init__(self, input_dim, output_dim, fit_intercept=True, n_iter=1000, tol=1e-5, C=0.01, **parms):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fit_intercept = fit_intercept
        if self.fit_intercept:
            self.W_ = np.random.randn(output_dim, input_dim + 1)
        else:
            self.W_ = np.random.randn(output_dim, input_dim)
        self.n_iter = n_iter
        self.tol = tol
        self.C = C
        self._cost_func = create_cost_func(sum_of_square_error, l2_penalty, self.C, self.fit_intercept)
        self._cost_grad_func = create_cost_grad_func(sum_of_square_error_grad, l2_penalty_grad, self.C,
                                                     self.fit_intercept)

    def fit(self, X, y):
        if self.fit_intercept:
            D = np.column_stack((X, np.ones(X.shape[0])))
        else:
            D = X

        W = self._fit(D, y)
        self.W_ = W.reshape(self.output_dim, W.size / self.output_dim)

        print("final error = {0}".format(self._cost_func(self.W_, D, y)))

        return self

    def _fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        if self.fit_intercept:
            D = np.column_stack((X, np.ones(X.shape[0])))
        else:
            D = X

        return D.dot(self.W_.T)


class SGDLinearRegression(_BaseLinearRegression):
    def __init__(self, input_dim, output_dim, fit_intercept=True, n_iter=1000, tol=1e-5, C=0.01, batch_size=100, lr=0.001, report=0,
                 momentum=0.9):
        """
        Linear Regression using Stocastic Gradient Descent(SGD)
        :param input_dim: int dimension of independent variable
        :param fit_intercept: boolean
            if True explicitly fit intercept, if False assume the independent variables include dummy ones
        :param n_iter: int, maximum iteration
        :param tol: float, error tolerance under which the iteration will stop
        :param C: float, L2 regularization parameter
        :param batch_size: int batch size
        :param lr: float learning ratio
        """
        _BaseLinearRegression.__init__(self, input_dim, output_dim, fit_intercept, n_iter, tol, C)
        self.lr = lr
        self.batch_size = batch_size
        self.report = report
        self.momentum = momentum

    def _fit(self, X, y):
        W, error, converged = sgd(self._cost_func, self._cost_grad_func, self.W_.flatten(), X, y, self.n_iter, self.lr,
                                  self.batch_size, self.tol, self.report, self.momentum)

        return W


class BatchLinearRegression(_BaseLinearRegression):
    def __init__(self, input_dim, output_dim, fit_intercept=True, n_iter=1000, tol=1e-5, C=0.01, method="CG"):
        """
        Linear Regression using Batch Gradient Descent such as Conjugate Gradient (CG) or BFGS
        :param input_dim: int dimension of independent variable
        :param fit_intercept: boolean
            if True explicitly fit intercept, if False assume the independent variables include dummy ones
        :param n_iter: int, maximum iteration
        :param tol: float, error tolerance under which the iteration will stop
        :param C: float, L2 regularization parameter
        :param method: string method name passes to scipy.optimize.minimize
        """
        _BaseLinearRegression.__init__(self, input_dim, output_dim, fit_intercept, n_iter, tol, C)
        self.method = method

    def _fit(self, X, y):
        result = minimize(self._cost_grad_func, self.W_.flatten(), (X, y), self.method, jac=True,
                          tol=self.tol,
                          options={"maxiter": self.n_iter})
        return W


class LinearRegression(_BaseLinearRegression):
    def __init__(self, input_dim, output_dim, fit_intercept=True, n_iter=1000, tol=1e-5, C=0.01, optimizer=sgd, batch_size=100,
                 lr=0.001, report=0, momentum=0.9, method="CG"):
        _BaseLinearRegression.__init__(self, input_dim, output_dim, fit_intercept, n_iter, tol, C)
        self.lr = lr
        self.batch_size = batch_size
        self.report = report
        self.momentum = momentum
        self.method = method
        self.optimizer = optimizer

    def _fit(self, X, y):
        W, error, converged = self.optimizer(self._cost_func, self._cost_grad_func, self.W_.flatten(), X, y,
                                             self.n_iter, tol=self.tol, lr=self.lr, batch_size=self.batch_size,
                                             report=self.report, momentum=self.momentum, method=self.method)
        return W
