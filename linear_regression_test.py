__author__ = 'du'

import numpy as np
from road_to_deep_learning.utils import optimize
from road_to_deep_learning.naive import LinearRegression

np.random.seed(0)
W = np.atleast_2d([2, -1])
X = np.random.randn(10000, 2)
y = X.dot(W.T) + 1 + np.random.randn(len(X), 1)

optimizer = optimize.sgd
model = LinearRegression(X.shape[1], y.shape[1], tol=1e-4, batch_size=200, n_iter=1000000, optimizer=optimize.scipy_minimize,
                         method="L-BFGS-B", report=100).fit(X, y)
print(model.W_)
print(model.score(X, y))
