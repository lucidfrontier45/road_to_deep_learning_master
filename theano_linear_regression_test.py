__author__ = 'du'

import numpy as np
from road_to_deep_learning.utils import optimize
from road_to_deep_learning.theano import LinearRegression

np.random.seed(0)
W = np.atleast_2d([[2, -1, 3], [-3, 4, 1]])
X = np.random.randn(10000, 3)
y = X.dot(W.T) + np.array([1, 2])
y += np.random.randn(*y.shape)

optimizer = optimize.sgd
model = LinearRegression(X.shape[1], y.shape[1], tol=1e-6, batch_size=200, n_iter=1000000, report=1000).fit(X, y)
print(model.get_variables())
print(model.score(X, y))
