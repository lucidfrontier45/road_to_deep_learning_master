__author__ = 'du'

import numpy as np
from sklearn import base
from scipy.optimize import minimize


def _linear_regression_loss_function(W, X, y, C=0):
    e = - (y - X.dot(W))
    grad = np.sum(e[:, np.newaxis] * X, 0)

    if C > 0:
        e += C * W.dot(W)
        grad += 2.0 * C * W

    return e.dot(e), grad


class _BaseLinearRegression(base.BaseEstimator, base.RegressorMixin):
    def __init__(self, n_dim, fit_intercept=True, n_iter=1000, eps=1e-5, C=0.01, **parms):
        self.n_dim = n_dim
        self.fit_intercept = fit_intercept
        if self.fit_intercept:
            self.W_ = np.random.randn(self.n_dim + 1)
        else:
            self.W_ = np.random.randn(self.n_dim)
        self.n_iter = n_iter
        self.eps = eps
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
    def __init__(self, n_dim, fit_intercept=True, n_iter=1000, eps=1e-5, C=0.01, batch_size=100, lr=0.001):
        _BaseLinearRegression.__init__(self, n_dim, fit_intercept, n_iter, eps, C)
        self.lr = lr
        self.batch_size = batch_size

    def _fit(self, X, y):

        N = len(y)
        error = 1e10
        for i in range(self.n_iter):
            if 0 < self.batch_size < N:
                idx = np.random.permutation(self.batch_size)
                XX = X[idx]
                yy = y[idx]
            else:
                XX = X
                yy = y
            e, grad = _linear_regression_loss_function(self.W_, XX, yy, self.C)
            self.W_ = self.W_ - self.lr * grad
            # if i % 100 == 0:
            #     print(i, e2, error - e2, self.eps)
            e = - (y - X.dot(self.W_))
            e2 = e.dot(e)
            if error - e2 < self.eps:
                print(i, e2, error - e2, self.eps)
                break
            error = e2


class BatchLinearRegression(_BaseLinearRegression):
    def __init__(self, n_dim, fit_intercept=True, n_iter=1000, eps=1e-5, C=0.01, method="CG"):
        _BaseLinearRegression.__init__(self, n_dim, fit_intercept, n_iter, eps, C)
        self.method = method

    def _fit(self, X, y):
        result = minimize(_linear_regression_loss_function, self.W_, (X, y, self.C), self.method, jac=True, tol=self.eps,
                          options={"maxiter": self.n_iter})
        self.W_ = result.x


if __name__ == "__main__":
    np.random.seed(0)
    W = [2, -1]
    X = np.random.randn(10000, 2)
    y = X.dot(W) + 1 + np.random.randn(len(X))

    model1 = SGDLinearRegression(X.shape[1], eps=1e-7, batch_size=100, n_iter=1000000).fit(X, y)
    print(model1.W_)
    print(model1.score(X, y))

    model2 = BatchLinearRegression(X.shape[1], eps=1e-7, n_iter=1000000, method="L-BFGS-B").fit(X, y)
    print(model2.W_)
    print(model2.score(X, y))
