__author__ = 'du'

import numpy as np
from road_to_deep_learning.utils import optimize, functions
from road_to_deep_learning.naive import MLPRegression
from sklearn import datasets

np.random.seed(0)
W = np.atleast_2d([[2, -1], [-3, 4]])
X = np.random.randn(10000, 2)
y = X.dot(W.T) + np.array([1, 2]) + np.random.randn(*X.shape)

input_dim = X.shape[1]
output_dim = y.shape[1]

model = MLPRegression(input_dim, 3, output_dim, batch_size=100, tol=1e-7, lr=0.001, optimizer=optimize.scipy_minimize,
                      n_iter=100000, report=10, lambda_input=0.0, lambda_hidden=0.0, method="L-BFGS-B",
                      activation=functions.Identity()).fit(X, y)
print(model.w_hidden.T.dot(model.w_input))
print(model.score(X, y))
