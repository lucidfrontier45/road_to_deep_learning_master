__author__ = 'du'

import numpy as np
from road_to_deep_learning.theano.mlp import MLPRegressor

np.random.seed(0)
W = np.atleast_2d([[2, -1], [-3, 4]])
X = np.random.randn(10000, 2)
y = X.dot(W.T) + np.array([1, 2]) + np.random.randn(*X.shape)

input_dim = X.shape[1]
output_dim = y.shape[1]

model = MLPRegressor(input_dim, 3, output_dim, tol=1e-6, lr=0.05, n_iter=100000, report=1000).fit(X, y)
print(model.get_variables())
print(model.score(X, y))
