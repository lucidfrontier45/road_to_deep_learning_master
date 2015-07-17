__author__ = 'du'

import numpy as np


def format_result(i, e, de, tol):
    return "{0:>8d}, {1:.4e}, {2:.4e}, {3:.4e}".format(i, e, de, tol)


def gd(objective_f, grad_f, W_init, X, y, C=0.01, n_iter=1000, lr=0.01, tol=1e-5, report=0):
    error = 1e10
    W = W_init.copy()
    converged = False
    for i in range(n_iter):
        _, grad = grad_f(W, X, y, C)
        W -= lr * grad / len(y)
        e = objective_f(W, X, y, C)
        if report > 0 and i % report == 0:
            print(format_result(i, e, error - e, tol))
        if tol > 0 and error - e < tol:
            print(format_result(i, e, error - e, tol))
            print("converged")
            converged = True
            break
        error = e

    return W, error, converged


def sgd(objective_f, grad_f, W_init, X, y, C=0.01, n_iter=1000, lr=0.01, batch_size=100, tol=1e-5, report=0):
    N = len(y)
    assert N > batch_size
    assert batch_size > 0
    n_batch = int(N / batch_size)
    error = 1e10
    W = W_init.copy()
    converged = False

    for i in range(n_iter):
        # shuffle data
        idx = np.random.permutation(N)
        XX = X[idx]
        yy = y[idx]

        for b in range(n_batch):
            start = b * batch_size
            stop = start + batch_size
            _, grad = grad_f(W, XX[start:stop], yy[start:stop], C)
            W -= lr * grad / batch_size

        e = objective_f(W, X, y, C)
        if report > 0 and i % report == 0:
            print(format_result(i, e, error - e, tol))
        if tol > 0 and abs(error - e) < tol:
            print(format_result(i, e, error - e, tol))
            print("converged")
            converged = True
            break
        error = e

    return W, error, converged
